# FMPC Stage 06 Split-Update Contract Family Planning Addendum

This file is a planning-only Stage 06 addendum.

It does not change the current implemented Stage 06 mathematics by itself.

Current implemented Stage 06 authority remains:

- [specs/stage_06_low_budget_efficiency.md](/e:/CodeSpace/PredictiveCoding/specs/stage_06_low_budget_efficiency.md)

This addendum exists only to define the next prospective Stage 06 family boundary after the
authoritative v1 baseline and the authoritative v2 follow-up have both been tested.

## 20. Stage 06 split-update contract family

### 20.1 Scope

This is a planning-only charter for the next Stage 06 family.

It does not yet:

- add a new implementation
- add a new experiment entry
- add a new test
- replace the current Stage 06 baseline artifact
- override the current Stage 06 low-budget gate in `validation.md`

### 20.2 Post-v2 reason for opening a new family

Directly confirmed from the current authoritative artifacts:

- `stage06_v1_objective_curriculum_energydrop_default`
  - passed Tier 1 viability
  - failed the Tier 2 main gate against the matched-budget Stage 05 control
  - did not justify a `512` rescue
- `stage06_v2_persistent_overlap_objective_curriculum_energydrop_default`
  - kept the same A2 family boundary
  - changed only the late objective schedule from hard handoff to persistent overlap
  - produced only a small raw mechanism improvement relative to v1
  - still failed the Tier 2 main gate
  - still did not justify a `512` rescue
  - had materially worse runtime proxy and worse mechanism gain per runtime than v1

Planning inference from those results:

- the remaining Stage 06 bottleneck is now more plausibly the current A2 aggregate contract itself
  rather than the exact late-phase schedule shape inside that contract
- the default next move should therefore not be another narrow `v2.x` schedule-retuning pass

### 20.3 Provisional family name

The next prospective Stage 06 family is provisionally named:

- `stage06_B1_split_update_contract_family`

The first planned probe inside this family is provisionally named:

- `stage06_v3_split_update_objective_contract_default`

### 20.4 One-sentence hypothesis

The current Stage 06 Tier 2 gap is more likely driven by destructive interference between
trajectory-side and semigroup-side supervision inside one simultaneous aggregate update than by
the exact late objective schedule, so separating those updates at the contract level may improve
matched-budget low-budget efficiency without reopening Stage 05 geometry work.

### 20.5 Single primary changed axis

The single primary changed axis is:

- replace the current simultaneous weighted-sum aggregate objective update path with a split-update
  contract over the same reused Stage 05 trajectory and semigroup targets

Interpretation:

- this is an update-path / supervision-coupling change
- this is not a numeric schedule retune
- this is not a return to Stage 05 v3-A branchwise supervision by default

### 20.6 What remains unchanged

Unless a later implementation pass explicitly justifies a wider change, this family keeps:

- the Stage 05 two-branch parameterization
- Stage 05 target-builder reuse
- aggregate residual supervision as the boundary condition for the family input/output surface
- `r` as remaining horizon
- `z_roll = z_t + r * u_psi(z_t, r, t; c)`
- the energy-drop and fixed-point penalty families as existing comparison context
- matched-budget control:
  - `stage05_v3c_stronger_semigroup_weight`
- the Stage 06 low-budget tier structure and hard gate

### 20.7 Low-budget-first and matched-budget rule

Any first implementation inside this family must remain:

- low-budget-first
- matched-budget
- evaluated first at:
  - `128 epochs`
  - `256 epochs`
- eligible for `512` only if the `256`-epoch result already shows a credible positive trend

Long-budget runs must not be used as the first existence proof.

### 20.8 Acceptance gate

The first probe in this family only earns continuation if it:

- stays mechanism-positive at low budget
- materially beats the matched-budget Stage 05 control at Tier 2
  - or shows clearly better cost-effectiveness under the existing Stage 06 efficiency record
- avoids an obvious report-only accuracy collapse

This file does not change the current authoritative gate in:

- [validation.md](/e:/CodeSpace/PredictiveCoding/validation.md)

### 20.9 Kill criteria

Default kill criteria for the first probe in this family:

- it still fails the Tier 2 main gate
- and it still does not show clearly better cost-effectiveness than the matched-budget Stage 05 control

If that happens, do not automatically open a second narrow same-family retune.

### 20.10 Artifact naming convention

If this family later enters implementation, use:

- comparison root:
  - `outputs/stage_06_low_budget_efficiency/stage06_v3_split_update_comparison/`
- candidate run root:
  - `outputs/stage_06_low_budget_efficiency/fmpc_stage06_objective_curriculum/`

### 20.11 Explicit non-goals

This family does not authorize by default:

- another `v2.x` schedule-retuning pass
- reopening the Stage 05 continuation / midpoint / coupled / precision / scaled micro-family
- restoring Stage 05 v3-A branchwise supervision as the default next move
- long-budget-first validation
- multiple parallel Stage 06 families

