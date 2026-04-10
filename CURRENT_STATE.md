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
- The adopted-package readout-alignment confirmation is sealed as a no-op:
  - `readout_align_final_w050` and `readout_align_every_w050` remain identical to the adopted control on validation accuracy, test accuracy, gate counts, transported energy, and report-output MSE
- Older bootstrap-target-side follow-ups are not credible remaining limiters from the current adopted-package state:
  - detached slow-PC bootstrap source did not materially beat the local-field source end-to-end
  - one-step lagged target snapshots did not improve behavior
  - no tested bootstrap↔identity curriculum materially beat the fixed-4-step corrective default
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
- The increment-only confirmation pass now says the preterminal live successor increment is still the blocker:
  - the narrow direction trust-region candidate toward the cached increment keeps much more of the failed earlier-control gain than the safe cached-increment lower bound:
    - validation-accuracy retention vs failed anchor: about `75.9%`
    - terminal row-space RMS retention vs failed anchor: about `65.8%`
  - but it still reopens too much of the gate collapse:
    - `seed_gate_positive_rate: 0.4`
    - `selected_epoch_passes_gate_rate: 0.4`
    - `selector_fallback_used_rate: 0.6`
  - the current confirmation-level diagnosis is:
    - `live_successor_increment_blocker_persists`
- The increment-internal source-localization pass now says the blocker sits mainly in the live successor-increment direction rather than in its magnitude:
  - `exact cached direction + live magnitude` restores the full selector/gate contract:
    - `seed_gate_positive_rate: 1.0`
    - `selected_epoch_passes_gate_rate: 1.0`
    - `selector_fallback_used_rate: 0.0`
    - but it collapses back to near-control accuracy and terminal row-space metrics
  - `cached magnitude + live direction` stays almost identical to the failed higher-gain unstable reference:
    - `mean_val_accuracy: 0.8570`
      - `mean_gate_passing_epoch_count: 0.0`
      - `selector_fallback_used_rate: 1.0`
  - the current increment-internal diagnosis is:
    - `live_successor_increment_direction_is_primary_blocker`
- The direction-only confirmation pass now says the live successor-increment direction is still blocked under the current selector/gate contract:
  - the weaker `45` degree trust-region keeps more of the failed earlier-control gain than the current `30` degree partial-signal reference:
    - validation-accuracy retention vs failed anchor: about `82.8%`
    - terminal row-space RMS retention vs failed anchor: about `86.3%`
    - but it fully reopens the gate collapse:
      - `seed_gate_positive_rate: 0.0`
      - `selector_fallback_used_rate: 1.0`
  - the stronger `20` degree trust-region partially recovers gate robustness:
    - `seed_gate_positive_rate: 0.8`
    - `selected_epoch_passes_gate_rate: 0.8`
    - `selector_fallback_used_rate: 0.2`
    - but it still does not keep the full selector/gate contract intact
  - the current direction-only confirmation diagnosis is:
    - `live_successor_increment_direction_blocker_persists`
- The direction-magnitude interaction pass now says cached-magnitude interaction does not materially improve on the current direction-only tradeoff:
  - `30` degree direction trust-region + cached magnitude remains effectively identical to the existing `30` degree direction-only reference:
    - validation-accuracy retention vs failed anchor: about `75.9%`
    - gate-robustness recovery vs control: about `25%`
    - terminal row-space RMS retention vs failed anchor: about `65.9%`
  - `20` degree direction trust-region + cached magnitude remains effectively identical to the existing `20` degree direction-only reference:
    - validation-accuracy retention vs failed anchor: about `44.8%`
    - gate-robustness recovery vs control: about `54.8%`
    - terminal row-space RMS retention vs failed anchor: about `42.0%`
  - neither interaction candidate keeps the full selector/gate contract intact
  - the current direction-magnitude interaction diagnosis is:
    - `live_successor_increment_interaction_blocker_persists`
- The deeper live successor-increment formulation pass now says the bad live direction is not yet localizable to a single additive internal term inside the residualized-local-field raw velocity:
  - swapping only the learned residual term to the cached analogue while keeping the live detached local-field anchor term leaves behavior effectively identical to the failed live/live earlier-control reference:
    - `mean_val_accuracy: 0.8570`
    - `mean_gate_passing_epoch_count: 0.0`
    - `selector_fallback_used_rate: 1.0`
    - `mean_val_terminal_rowspace_rms: 0.1425`
  - swapping only the detached local-field anchor term to the cached analogue while keeping the live learned residual term also leaves behavior effectively identical to the failed live/live earlier-control reference:
    - `mean_val_accuracy: 0.8570`
    - `mean_gate_passing_epoch_count: 0.0`
    - `selector_fallback_used_rate: 1.0`
    - `mean_val_terminal_rowspace_rms: 0.1425`
  - neither single-term substitution restores any gate robustness, and both retain essentially `100%` of the failed-anchor accuracy / row-space gain
  - the current formulation-level diagnosis is:
    - `bad_live_direction_source_not_yet_localized_but_formulation_blocker_strengthened`
- The remaining-issue triage pass now says no different package-internal issue survives as a credible next adopted-package diagnostic:
  - the readout-alignment family is sealed as an exact no-op on the adopted package
  - the older bootstrap-source, target-lag, and curriculum families do not show a credible reopen path from the current adopted-package state
  - combined with the strengthened live successor-increment formulation blocker, the adopted corrective TF2 package is now best treated as locally saturated under the current selector/gate contract

## Current Recommendation

- Current next narrow move:
  - stop package-internal TF2 digging from this state
  - only reopen TF2 mainline digging if a genuinely different issue family appears from new evidence or if the project explicitly decides to leave the current package / selector-gate contract
- Current active question:
  - no different remaining package-internal issue is currently recommended for pursuit inside the adopted TF2 package
- Post-triage decision memo:
  - package-internal TF2 digging is considered closed from the current state because the remaining slow-PC gap is still present, but every credible internal corrective package issue family now has a sealed negative result:
    - readout alignment is an exact no-op on the adopted package
    - detached bootstrap-source, one-step target-lag, and bootstrap-to-identity curriculum follow-ups do not materially improve the adopted package
    - the late-rollout successor-value and successor-increment line now terminates in a strengthened formulation-level blocker rather than an adoption-viable local fix
  - the recommended next project move is:
    - keep the current adopted TF2 package frozen as the bridge result on `main`
    - redirect the next planning effort to a post-TF2 teacher-free FMPC / EF exploratory line that explicitly leaves the current corrective package or selector-gate contract
    - do not rewrite the current active line on `main` until that exploratory line is explicitly chartered

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
  - [outputs/fmpc_tf2_readout_alignment_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_readout_alignment_suite)
  - [outputs/fmpc_tf2_bootstrap_source_bias_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_bootstrap_source_bias_suite)
  - [outputs/fmpc_tf2_target_lag_coupling_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_target_lag_coupling_suite)
  - [outputs/fmpc_tf2_curriculum_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_curriculum_suite)
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
  - [outputs/fmpc_tf2_successor_increment_confirmation_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_successor_increment_confirmation_suite)
  - [outputs/fmpc_tf2_successor_increment_source_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_successor_increment_source_suite)
  - [outputs/fmpc_tf2_successor_increment_direction_confirmation_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_successor_increment_direction_confirmation_suite)
  - [outputs/fmpc_tf2_successor_increment_interaction_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_successor_increment_interaction_suite)
  - [outputs/fmpc_tf2_successor_increment_formulation_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_successor_increment_formulation_suite)

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
