# FMPC Stage 06 Low-Budget Efficiency Addendum

This file extends the baseline math in [spec_math.md](/e:/CodeSpace/PredictiveCoding/spec_math.md),
the Stage 03 transport addendum in
[specs/stage_03_transport_core_v1.md](/e:/CodeSpace/PredictiveCoding/specs/stage_03_transport_core_v1.md),
and the Stage 05 EF core probe addendum in
[specs/stage_05_ef_core_probe.md](/e:/CodeSpace/PredictiveCoding/specs/stage_05_ef_core_probe.md).

To build a complete mathematical understanding for Stage 06 work, read in this order:

1. [spec_math.md](/e:/CodeSpace/PredictiveCoding/spec_math.md)
2. [specs/stage_03_transport_core_v1.md](/e:/CodeSpace/PredictiveCoding/specs/stage_03_transport_core_v1.md)
3. [specs/stage_05_ef_core_probe.md](/e:/CodeSpace/PredictiveCoding/specs/stage_05_ef_core_probe.md)
4. this addendum

Scope and precedence:

- this addendum overrides the Stage 05 addendum only inside the explicit Stage 06 low-budget efficiency scope
- outside that scope, the baseline, Stage 03, and Stage 05 addenda remain authoritative
- this addendum does not reopen Stage 04 package-internal work
- this addendum now defines the first implemented Stage 06 objective-contract line
- this addendum remains narrow and does not by itself reopen a broad Stage 06 search space

## 19. Stage 06 Low-Budget Efficiency addendum

Stage 06 opens a new efficiency-first charter on top of the Stage 05 mechanism evidence base.

Stage 06 is:

- a low-budget and low-compute viability stage
- a matched-budget efficiency stage
- scaffold-preserving by default at the level of two-branch parameterization and Stage 05 target-builder reuse
- mechanism-first, but no longer budget-indifferent

Stage 06 is not:

- a repudiation of Stage 05
- another narrow continuation / midpoint / coupled / precision / scaled pass inside the saturated v3-C micro-family
- a long-budget existence-proof stage
- a claim that Stage 05 or Stage 06 replaces the frozen Stage 04 bridge result on `main`

### 19.1 Relationship to Stage 05

Stage 05 remains important, but its role changes at Stage 06.

- Stage 05 is frozen as the mechanism-first exploration stage
- `stage05_v3c_stronger_semigroup_weight` remains the current high-budget Stage 05 mechanism reference
- `stage05_v3c_stronger_semigroup_weight` is also the current matched-budget Stage 05 control for Stage 06 comparisons
- the fixed-budget Stage 05 v2 result remains contextual evidence about budget sensitivity and efficiency limits, not the current Stage 06 matched-budget control
- the narrow v3-C contract-consolidation micro-family is treated as locally saturated and is excluded from default continuation

Interpretation:

- Stage 05 answered the question of whether the residual transport line contains a real mechanism signal
- Stage 06 asks whether that signal can survive under materially lower budget and lower compute conditions

### 19.2 Stage 06 problem statement

The main Stage 06 question is no longer:

- whether the Stage 05 mechanism exists at all

The main Stage 06 question is now:

- whether a scaffold derived from the validated Stage 05 line can remain mechanism-positive and economically credible under matched-budget low-budget conditions

Stage 06 therefore does not accept a candidate whose main proof of value appears only after:

- `1536` epochs
- `3072` epochs
- or another long-budget rescue that was not already justified by low-budget evidence

### 19.3 Preserved scaffold boundary

Stage 06 keeps the current validated scaffold boundary unless a later charter says otherwise.

Preserved by default:

- artifact-independent target construction
- the Stage 05 two-branch parameterization rooted in the v3-A explicit transport-drift decomposition:
  - `u_psi(z_t, r, t; c) = g_t + q_psi + d_psi`
- the trajectory-level scaffold validated through Stage 05 v3-B
- the endpoint / semigroup-consistency scaffold validated through Stage 05 v3-C
- deterministic artifact generation

Not preserved by default inside Stage 06 v1:

- the Stage 05 v3-A branchwise transport / drift supervision contract itself
- a requirement that Stage 06 update the two learned branches with explicit per-branch transport targets and drift targets

Not preserved as default search directions:

- another narrow semigroup-internalized geometry micro-variant inside the saturated v3-C consolidation line
- another pure budget push as the primary existence proof
- another pure same-family schedule tweak as the primary efficiency story

### 19.4 Stage 06 initial hypothesis space

The default Stage 06 hypothesis space is:

- objective curriculum
- update scheduling
- shorter or more direct error / transport pathways
- matched-budget efficiency contracts

The intended move is to work above the saturated Stage 05 geometry micro-family rather than inside it.

### 19.5 Minimal Stage 06 objective-mix notation

Stage 06 introduces a new high-level objective-mix coefficient:

- `beta_obj(k) in [0, 1]`

Interpretation:

- `beta_obj(k)` is an optimization-contract or objective-curriculum coefficient indexed by training step or epoch `k`
- `beta_obj(k)` is distinct from the Stage 05 split-horizon geometry parameter `alpha`
- `alpha` continues to refer to the trajectory split geometry when Stage 05 scaffold terms are reused
- `beta_obj(k)` controls objective emphasis or update emphasis, not trajectory geometry

The minimal Stage 06 abstraction is therefore:

- keep the Stage 05 transport family and target geometry scaffold fixed unless a later Stage 06 charter explicitly changes them
- let the next Stage 06 probe vary only a higher-level objective or update contract through `beta_obj(k)` and matched-budget scheduling

The first implemented Stage 06 line is now fixed as:

- `stage06_v1_objective_curriculum_energydrop_default`

It preserves the validated Stage 05 two-branch parameterization and target-builder scaffold, and changes only the training contract.

Normative implementation contract:

- keep the Stage 05 transport output notation:
  - `u_psi(z_t, r, t; c)`
- keep the Stage 05 two-branch parameterization:
  - `u_psi(z_t, r, t; c) = g_t + q_psi + d_psi`
- keep the Stage 05 local energy flow:
  - `g_t = -grad_z E_theta(z_t; c)`
- keep the Stage 05 trajectory-side target semantics as:
  - `L_traj`
- keep the Stage 05 semigroup / endpoint-consistency target semantics as:
  - `L_semi`
- keep Stage 05 target-builder reuse for those targets
- do not claim that Stage 06 v1 preserves the Stage 05 v3-A branchwise supervision contract:
  - Stage 06 v1 keeps the two-branch parameterization
  - but it trains those branches through an aggregate residual objective over reused Stage 05 targets
- add the one-step energy-drop penalty:
  - `r` keeps the Stage 03 / Stage 05 meaning of remaining horizon
  - `z_roll = z_t + r * u_psi(z_t, r, t; c)`
  - `L_drop = mean(relu(E_theta(z_roll; c) - E_theta(z_t; c) + delta_margin))`
- add the fixed-point contraction penalty:
  - `g_roll = -grad_z E_theta(z_roll; c)`
  - `L_fp = mean(||g_roll||_2^2)`

The Stage 06 v1 total loss is:

- `L_6A(k) = (1 - beta_obj(k)) * L_traj + beta_obj(k) * L_semi + lambda_energy_drop * L_drop + lambda_fixed_point * L_fp`

Default coefficients:

- `beta_obj_warmup_fraction = 0.25`
- `beta_obj_ramp_fraction = 0.50`
- final plateau fraction = `0.25`
- `lambda_energy_drop = 0.25`
- `lambda_fixed_point = 0.10`
- `delta_margin = 0.0`

The default Stage 06 `beta_obj(k)` schedule is:

- `beta_obj(k) = 0` for `k < 0.25K`
- linearly ramp from `0` to `1` over `[0.25K, 0.75K)`
- `beta_obj(k) = 1` for `k >= 0.75K`

Interpretation:

- `beta_obj(k)` remains distinct from the Stage 05 geometry parameter `alpha`
- Stage 06 v1 preserves the Stage 05 two-branch parameterization and target-builder scaffold
- Stage 06 v1 does not preserve the Stage 05 v3-A branchwise transport / drift supervision contract
- Stage 06 v1 does not continue the narrow Stage 05 v3-C midpoint / continuation micro-family

Current narrow A2 follow-up implementation surface:

- `stage06_v2_persistent_overlap_objective_curriculum_energydrop_default`

This follow-up remains inside the same Stage 06 A2 family:

- it keeps the Stage 05 two-branch parameterization
- it keeps Stage 05 target-builder reuse
- it keeps aggregate residual supervision
- it keeps `r` as remaining horizon
- it keeps:
  - `z_roll = z_t + r * u_psi(z_t, r, t; c)`
- it keeps the same energy-drop and fixed-point penalty families

Its only changed axis is the late objective schedule contract:

- it does not end in a pure `L_semi` plateau
- instead, it keeps both trajectory-side and semigroup-side objective weights non-zero in the late phase

The default v2 overlap schedule is:

- `beta_obj(k) = 0` for `k < 0.25K`
- linearly ramp from `0` to `0.75` over `[0.25K, 0.75K)`
- `beta_obj(k) = 0.75` for `k >= 0.75K`

Equivalent late-phase weights:

- trajectory-side objective weight:
  - `1 - beta_obj = 0.25`
- semigroup-side objective weight:
  - `beta_obj = 0.75`

Interpretation:

- Stage 06 v2 changes the objective schedule, not the Stage 06 family boundary
- Stage 06 v2 does not restore Stage 05 v3-A branchwise supervision
- Stage 06 v2 is an implemented follow-up surface, not an artifact-backed replacement for the current Stage 06 v1 baseline until a dedicated comparison artifact exists

### 19.6 Minimal Stage 06 acceptance framing

Stage 06 keeps mechanism metrics primary, but it changes the budget contract.

Normative framing:

- a candidate must show viability first at low budget
- long-budget runs can no longer serve as the first proof that a candidate is worth pursuing
- long-budget runs may still be used later as:
  - upper-bound validation
  - or matched-budget contextual controls

The operational budget tiers and hard gates live in
[validation.md](/e:/CodeSpace/PredictiveCoding/validation.md).

### 19.7 Excluded default search space

Unless a later Stage 06 charter explicitly reopens them, Stage 06 does not continue by default with:

- `stage05_v3c_endpoint_line_midpoint_trajectory_contract`
- `stage05_v3c_endpoint_line_continuation_blend_trajectory_contract`
- `stage05_v3c_coupled_defect_projection_trajectory_contract`
- `stage05_v3c_precision_weighted_continuation_corrector_trajectory_contract`
- `stage05_v3c_scaled_continuation_blend_trajectory_contract`

Interpretation:

- those candidates remain part of the Stage 05 evidence chain
- they are still useful as mechanism lessons
- they are no longer the default next search space

### 19.8 Required Stage 06 implementation interpretation

The first implemented Stage 06 pass is framed as:

- a matched-budget, low-budget-first probe
- above the saturated Stage 05 v3-C geometry micro-family
- most likely inside an objective-curriculum or optimization-contract hypothesis space

It should not be framed as:

- another narrow Stage 05 geometry refinement
- another continuation-strength variant
- another midpoint / continuation / coupled local corrector variant
- another long-budget proof-of-existence pass

The current Stage 06 code-backed path is:

- implementation:
  - [src/pc/stage_06_low_budget_efficiency/fmpc_stage06_objective_curriculum.py](/e:/CodeSpace/PredictiveCoding/src/pc/stage_06_low_budget_efficiency/fmpc_stage06_objective_curriculum.py)
- experiment entry:
  - [experiments/stage_06_low_budget_efficiency/fmpc_stage06_objective_curriculum.py](/e:/CodeSpace/PredictiveCoding/experiments/stage_06_low_budget_efficiency/fmpc_stage06_objective_curriculum.py)
- low-budget comparison entry:
  - [experiments/stage_06_low_budget_efficiency/stage06_v1_low_budget_comparison.py](/e:/CodeSpace/PredictiveCoding/experiments/stage_06_low_budget_efficiency/stage06_v1_low_budget_comparison.py)
