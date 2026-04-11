# Stage 05 V2 Diagnostics

## Scope
- source comparison: `outputs/stage_05_ef_core_probe/frozen_bridge_vs_two_branch_corrected_core_comparison`
- seeds: `[0, 1, 2]`
- this pass keeps Stage 04 frozen and keeps the Stage 05 v2 transport family unchanged

## Training Boundary
- selected epochs: `[12, 12, 12]`
- total epochs: `[12, 12, 12]`
- selection hits final training boundary on all seeds: `True`

## Epoch-Level Diagnosis
- configured-step energy still improving near final epoch rate: `1.0`
- configured-step residual still improving near final epoch rate: `1.0`
- validation accuracy still improving near final epoch rate: `1.0`
- final-epoch mean validation accuracy: `0.270370`

## Branch Contribution
- state branch contribution label: `clearly_material`
- mean `||m_state|| / ||m_traj||`: `0.703646`
- mean state-term / trajectory-term ratio: `0.000154`
- forward state contribution material but identity-term contribution small: `True`

## Rollout Diagnosis
- rollout gap label: `meaningful_extra_gain_after_step1`
- mean one-step energy delta vs identity: `-0.000172769874`
- mean configured-step energy delta vs identity: `-0.000176362138`
- mean incremental configured-step energy gain over step 1: `-0.000081609787`

## Selection-Rule Diagnosis
- best-accuracy epoch is selected-epoch rate: `1.0`
- mean selected-vs-best accuracy gap: `0.000000`
- selection rule likely primary issue: `False`

## Final Diagnosis
- selected label: `likely_undertrained`
- rationale: `All Stage 05 v2 seeds select the final training epoch and configured-step mechanism plus validation accuracy are still improving at the training boundary.`
- next Stage 05 v3 target: `longer_training_or_budget`
