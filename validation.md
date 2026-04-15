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

## Stage 05 Validation Status

Stage 05 is now a frozen mechanism-reference stage, not the default forward efficiency line.

Current Stage 05 evidence that matters for validation framing:

- the `1536 -> 3072` budget push on the Stage 05 v2 line materially improved configured-step mechanism and still selected the final epoch on every seed
- the same-family fixed-`1536` efficiency tweak was effectively a no-op and did not materially narrow the gap to the contextual `3072` reference
- the refined v3-C recompare materially established `stage05_v3c_stronger_semigroup_weight` above promoted refined v3-B on configured-step mechanism
- the continuation / midpoint / coupled / precision / scaled v3-C micro-family then remained directionally useful but locally saturated under the current threshold

Operational interpretation:

- Stage 05 answered the existence question for mechanism signal
- Stage 05 did not answer the efficiency question
- Stage 05 artifacts remain valid as mechanism reference evidence, but not as automatic efficiency wins

## Stage 06 Low-Budget Efficiency Validation Contract

Stage 06 changes the main validation question.

The new question is:

- can a candidate remain mechanism-positive and economically credible under matched-budget low-budget conditions

The new question is not:

- can a candidate eventually look good if the budget is pushed high enough

### Stage 06 Budget Tiers

- Tier 1:
  - `128 epochs`
  - quick viability screen only
- Tier 2:
  - `256 epochs`
  - main gate
- Tier 3:
  - `512 epochs`
  - one-time rescue only for edge candidates that already show a credible positive trend at Tier 2

### Stage 06 Hard Gate

The hard Stage 06 gate is:

1. every new candidate must first run under the low-budget gate
2. `256 epochs` is the main decision tier
3. if a candidate does not show a clear shared-seed mechanism-positive signal at `256 epochs`, it fails by default
4. if a candidate does not show stable matched-budget pairwise gain or clearly better cost-effectiveness at `256 epochs`, it fails by default
5. `512 epochs` is allowed only once, and only when the `256`-epoch result already shows a consistent positive trend
6. `1536+` may not be used as the first existence proof before the low-budget gate is passed
7. long-budget runs may only serve later as:
   - upper-bound checks
   - or matched-budget contextual controls

### Stage 06 Gate Metrics

Primary gate items:

- low-budget one-step mechanism-positive
- low-budget configured-step mechanism-positive
- matched-budget configured-step energy delta
- matched-budget configured-step residual delta
- no obvious report-only accuracy collapse

Interpretation:

- a candidate should not pass Stage 06 just because it has a small task-accuracy fluctuation
- a candidate should not pass Stage 06 just because it remains mechanism-positive while being much more expensive

### Stage 06 Efficiency Record Metrics

These are now formal comparison fields, not optional side notes:

- `runtime_proxy_seconds`
- mechanism gain per runtime
- mechanism gain per epoch

Interpretation:

- `runtime_proxy_seconds` must be reported in every real Stage 06 comparison
- it is not the only hard gate
- but a stronger mechanism result with a clearly worse cost profile does not count as an automatic success

### Stage 06 Success Standard

A Stage 06 candidate is only a credible forward result if it simultaneously:

- stays mechanism-positive at low budget
- avoids an obvious report-only accuracy collapse
- shows a stable matched-budget configured-step gain over the relevant reference
  - or shows clearly better cost-effectiveness when raw mechanism gain is similar
- survives the low-budget gate before any long-budget contextual run is considered

### Stage 06 Default Failure Interpretation

Default failure cases now include:

- no stable positive signal by `256` epochs
- mechanism signal that appears only after a long-budget rescue
- a candidate that is slightly stronger on mechanism but obviously worse on total cost
- another narrow Stage 05 v3-C geometry micro-variant being presented as if it were the new main efficiency line

## Stage 05 Historical Exploratory Validation Contract

Historical note:

- the Stage 05 exploratory validation contract below remains important as the evidence record that opened Stage 06
- it should not be read as the current forward efficiency gate

Stage 05 was mechanism-first.

Primary acceptance metrics:

- one-step energy decrease relative to identity or no-transport
- few-step fixed-point residual decrease relative to identity or no-transport
- deterministic artifact generation
- artifact-independent target construction

Secondary report-only metrics:

- validation accuracy
- test accuracy

What this meant operationally:

- task accuracy was not the Stage 05 gate
- a weak task metric did not invalidate a probe if mechanism metrics were positive
- a strong task metric alone did not justify promotion without a clear mechanism advantage

## Current Recommended Validation Move

The current validation move is no longer another narrow Stage 05 geometry refinement.

The current validation move is:

- keep Stage 04 frozen as the implemented bridge result on `main`
- keep Stage 05 frozen as the high-budget mechanism reference stage
- keep `stage05_v3c_stronger_semigroup_weight` as the current high-budget Stage 05 mechanism reference
- use `stage05_v3c_stronger_semigroup_weight` as the current matched-budget Stage 05 control for Stage 06 comparisons
- treat the narrow v3-C contract-consolidation micro-family as locally saturated
- keep Stage 06 under the low-budget gate above
- treat `stage06_v1_objective_curriculum_energydrop_default` as the first real Stage 06 baseline artifact:
  - it passed Tier 1 viability
  - it failed the Tier 2 main gate against the matched-budget Stage 05 control `stage05_v3c_stronger_semigroup_weight`
  - it did not justify a `512` rescue
  - the current authoritative baseline is the post-semantic-alignment rerun, not the older pre-alignment artifact
- require any later Stage 06 follow-up to remain:
  - matched-budget
  - low-budget-first
  - cost-aware
  - outside the continuation / midpoint / coupled / precision / scaled micro-family
  - materially different from the current Stage 06 v1 contract if reopened

Relevant active reference artifacts remain:

- `outputs/stage_05_ef_core_probe/stage05_v2_budget_push_validation_1536_to_3072/`
- `outputs/stage_05_ef_core_probe/stage05_v2_efficiency_diagnostic_at_1536/`
- `outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_refined_v3c_fixed_budget_recompare/`
- `outputs/stage_05_ef_core_probe/stage05_v3c_continuation_strength_diagnostic/`

## Relevant Active Artifacts

- frozen Stage 04 authority artifact tree:
  - `outputs/stage_04_incremental_bridge/`
- Stage 05 budget-push validation:
  - `outputs/stage_05_ef_core_probe/stage05_v2_budget_push_validation_1536_to_3072/`
- Stage 05 efficiency diagnostic:
  - `outputs/stage_05_ef_core_probe/stage05_v2_efficiency_diagnostic_at_1536/`
- Stage 05 refined v3-C recompare:
  - `outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_refined_v3c_fixed_budget_recompare/`
- Stage 05 continuation-strength diagnostic:
  - `outputs/stage_05_ef_core_probe/stage05_v3c_continuation_strength_diagnostic/`
- Stage 06 authoritative post-semantic-alignment baseline:
  - `outputs/stage_06_low_budget_efficiency/stage06_v1_low_budget_comparison/stage06_v1_post_semantic_alignment_rebaseline/`

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
