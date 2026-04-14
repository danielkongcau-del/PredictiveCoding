# FMPC Stage 05 EF Core Probe Addendum

This file extends the baseline math in [spec_math.md](/e:/CodeSpace/PredictiveCoding/spec_math.md)
and the Stage 03 transport addendum in
[specs/stage_03_transport_core_v1.md](/e:/CodeSpace/PredictiveCoding/specs/stage_03_transport_core_v1.md).

To build a complete mathematical understanding for Stage 05 work, read in this order:

1. [spec_math.md](/e:/CodeSpace/PredictiveCoding/spec_math.md)
2. [specs/stage_03_transport_core_v1.md](/e:/CodeSpace/PredictiveCoding/specs/stage_03_transport_core_v1.md)
3. this addendum

Scope and precedence:

- this addendum overrides the Stage 03 addendum only inside the explicit Stage 05 EF Core Probe scope
- outside that scope, the baseline and Stage 03 addendum remain authoritative
- this addendum does not silently redefine the baseline predictive-coding energy, the baseline hidden-state gradient, the baseline local parameter-update equations, or the slow iterative predict/eval path

## 18. Stage 05 EF Core Probe addendum

This addendum defines the first post-bridge, mechanism-first residual MeanFlow core validation stage.

Stage 05 is:

- a post-bridge theory-completion stage
- a residual average-velocity core validation stage
- artifact-independent in target construction
- mechanism-first in evaluation

Stage 05 is not:

- a reopening of Stage 04 package-internal stabilizer work
- a migration of terminal angle clipping, row-space splitting, or transported-output alignment into the new core family
- a scaling or topology stage
- a teacher-target stage

### 18.1 Scope

Stage 05 stays on the current layered predictive-coding substrate.

- it does not define a new substrate class
- it does not redefine predict-mode inference
- it does not introduce teacher trajectories, teacher fixed points, or teacher-generated regression targets
- it does not promote a new task-accuracy gate at this exploratory stage

### 18.2 Training context and hidden latent

Stage 05 uses the same supervised training context as Stage 03:

- `c = (x, y)`
- `x^0 = x` remains clamped
- `x^L = y` remains clamped

The hidden latent remains:

- `z = flatten(x^1, ..., x^(L-1))`

Only the free hidden layers are included in `z`.

### 18.3 Exact local flow

Stage 05 reuses the exact local flow:

- `g_theta(z_t; c) = -∇_z E_theta(z_t; c)`

where `E_theta` is still the baseline predictive-coding energy reconstructed from the layered state list.

### 18.4 Residual MeanFlow transport family

Stage 05 uses a residual average-velocity family:

- `u_psi(z_t, r, t; c) = g_t + m_psi(z_t, r, t; c)`

where:

- `g_t := g_theta(z_t; c)`
- `m_psi` is the learned residual average-velocity correction

The minimal residual input contract is fixed to:

- `m_psi_input = concat([z_t, target_onehot, t, r])`

Stage 05 keeps this minimal contract in v1:

- no appended current-state feature blocks
- no Stage 04 stabilizer family
- no scaling/topology extension

### 18.4.1 Stage 05 v2 two-branch residual core

Stage 05 v2 keeps the same residual MeanFlow family but replaces the single residual head with
an explicit two-branch decomposition:

- `m_psi = m_traj + m_state`

The two branch input contracts are:

- `m_traj_input = concat([z_t, target_onehot, t, r])`
- `m_state_input = concat([g_t, e_out_t, F_t])`

Interpretation:

- `m_traj` handles transport-coordinate prediction
- `m_state` handles current-state geometric correction

The transported average velocity remains:

- `u_psi(z_t, r, t; c) = g_t + m_traj(z_t, target_onehot, t, r; c) + m_state(g_t, e_out_t, F_t; c)`

This is still a Stage 05 residual MeanFlow core:

- it is not a new training family
- it does not migrate Stage 04 stabilizers
- it does not challenge the frozen Stage 04 bridge result on `main`

### 18.5 Bootstrap residual target

Stage 05 keeps the Stage 03 local self-bootstrap average-velocity anchor:

- `u_boot(z_t, r, t; c) = (Phi_LF_r(z_t; c) - z_t) / r`

The residual bootstrap target is:

- `m_boot = u_boot - g_t`

This target remains artifact-independent.

### 18.6 Corrected residual fixed-terminal-time identity

Stage 03 uses the fixed-terminal-time identity:

- `u ≈ g_t + r * D_T u`

For the residual family `u = g + m`, Stage 05 rewrites this as:

- `m ≈ r * D_T g_t + r * D_T m`

Stage 05 therefore defines the residual identity target as:

- `m_id = r * D_T g_t + r * D_T m_psi`

This is the normative Stage 05 correction relative to a residual-only JVP approximation.

### 18.6.1 Stage 05 directional-derivative approximations

Under the current minimal input contract:

- `g_t` does not explicitly depend on `t` or `r`
- `m_psi` depends on `z_t`, `target_onehot`, `t`, and `r`

Stage 05 approximates the two derivative terms as follows.

Anchor derivative:

- `D_T g_t ≈ [g(z_t + eps * g_t) - g(z_t - eps * g_t)] / (2 * eps)`

Residual-network derivative:

- `D_T m_psi` is approximated by a residual-network JVP with input tangent
- `[g_t, 0_target, +1_t, -1_r]`

The Stage 05 corrected residual identity target is then:

- `m_id = r * D_T g_t + r * D_T m_psi`

### 18.6.2 Stage 05 v2 branchwise corrected identity

Stage 05 v2 keeps the anchor derivative explicit and keeps the residual decomposition explicit.

For the two-branch residual core:

- `m_psi = m_traj + m_state`

the corrected residual identity is defined branchwise as:

- `m_id = r * D_T g_t + r * D_T m_traj + r * D_T m_state`

The required directional-derivative approximations are:

- anchor derivative:
  - `D_T g_t`
  - keep the centered finite-difference approximation along `g_t`
- trajectory-branch derivative:
  - `D_T m_traj`
  - compute by JVP on `m_traj_input = concat([z_t, target_onehot, t, r])`
  - use tangent `[g_t, 0_target, +1_t, -1_r]`
- current-state branch derivative:
  - `D_T m_state`
  - compute by JVP on `m_state_input = concat([g_t, e_out_t, F_t])`
  - use tangent `[D_T g_t, D_T e_out_t, D_T F_t]`

This keeps the Stage 05 v2 identity target mathematically explicit:

- anchor term
- trajectory term
- state term

The implementation must not silently collapse those three terms back into one opaque residual-only derivative.

### 18.7 Stage 05 v1 training objective

Stage 05 v1 uses:

- `L_boot = ||m_psi - m_boot||^2`
- `L_id = ||m_psi - m_id||^2`
- `L_total(k) = L_boot + lambda_id(k) * L_id`

This keeps the bootstrap anchor and the corrected residual identity inside one residual-output family.

### 18.8 Curriculum and training schedule

Stage 05 uses a three-phase training schedule:

- `warmup`
  - `lambda_id = 0`
  - theta update still uses the local-field rollout
- `transition`
  - theta update switches to the learned rollout
  - `lambda_id` follows a deterministic sigmoid ramp toward `lambda_id_max`
- `hybrid`
  - theta update continues to use the learned rollout
  - `lambda_id = lambda_id_max`

This adopts only the curriculum idea, not the full AlphaFlow time formula.

### 18.9 Parameter updates after transport

Stage 05 still changes only the training-time hidden-state transport path.

After transport produces a terminal hidden state `z_hat`, the repository still applies the same local parameter-update rule already defined in the baseline:

1. reconstruct the full state list from `z_hat` and the clamped context
2. recompute cache terms
3. compute baseline local parameter gradients
4. apply the same explicit parameter descent update

Stage 05 does not redefine the slow iterative predict/eval path.

### 18.10 Evaluation contract

Stage 05 remains mechanism-first.

Primary acceptance signals:

- one-step energy decrease relative to identity / no-transport
- few-step fixed-point residual decrease relative to identity / no-transport
- deterministic artifact generation
- no teacher dependency in target construction

Secondary report-only signals:

- validation accuracy
- test accuracy

Task accuracy is not the gate for Stage 05 v1.

The same mechanism-first evaluation contract remains in force for Stage 05 v2.

### 18.11 Stage 05 v3-A charter

This section opens the next Stage 05 charter only.

It does not itself modify:

- the current Stage 05 v2 implementation
- the current Stage 05 experiment entrypoints
- the current Stage 05 smoke or unit tests

Current evidence motivating this charter:

- Stage 05 v2 already shows positive one-step mechanism signal
- the main remaining weakness is configured-step mechanism efficiency
- same-family budget pushes still improve configured-step mechanism and report-only accuracy
- the fixed-budget same-family efficiency diagnostic no longer gives a material gain
- the next credible change should therefore target the higher-level mechanism contract rather than another schedule, epoch, or feature-flag tweak

#### 18.11.1 Working hypothesis

Stage 05 v3-A is motivated by the working hypothesis that the current residual target entangles
transport residual and anchor-drift residual, limiting configured-step efficiency.

This is a chartering hypothesis, not a proved repository conclusion.

The purpose of v3-A is to make that hypothesis explicit enough to test in the next implementation
pass.

#### 18.11.2 Explicit transport-drift decomposition

Under the current Stage 05 remaining-horizon notation:

- `t` is the current knot time
- `r` is the remaining horizon

Define the true average velocity and the average local flow over the same interval by:

- `u^*_{t,r} = (1 / r) * integral_t^{t+r} v_tau d tau`
- `gbar^*_{t,r} = (1 / r) * integral_t^{t+r} g_tau d tau`

When the normalized terminal-time interpretation `t + r = 1` is used, these may be written
equivalently over `[t, 1]`.

The current residual target can then be decomposed as:

- `b^*_{t,r} = u^*_{t,r} - g_t = (u^*_{t,r} - gbar^*_{t,r}) + (gbar^*_{t,r} - g_t)`

Interpretation:

- `u^*_{t,r} - gbar^*_{t,r}` is the transport residual
- `gbar^*_{t,r} - g_t` is the anchor drift or anchor mismatch

The v3-A working hypothesis is that the current high-level residual contract asks one learned
residual object to absorb both terms at once, which can preserve one-step value while releasing
configured-step efficiency too weakly.

#### 18.11.3 v3-A contract definition

Stage 05 v3-A is defined as an explicit transport-drift contract:

- `u_psi = g_t + q_psi + d_psi`

where:

- `q_psi` is the transport-residual branch
- `d_psi` is the anchor-drift branch

The bootstrap supervision is correspondingly decomposed into:

- `gbar_boot = average local flow along the same bootstrap rollout interval`
- `d_boot = gbar_boot - g_t`
- `q_boot = u_boot - gbar_boot`

This charter only fixes the high-level supervision semantics.

It does not yet fix:

- the exact v3-A loss weighting
- the final rollout-consistency terms
- the exact branch parameterization

Those choices belong to the next implementation pass.

#### 18.11.4 First repository implementation of v3-A

The first repository implementation of Stage 05 v3-A is intentionally minimal.

It reuses the current two-branch Stage 05 v2 scaffold and interprets it as:

- `u_psi = g_t + q_psi + d_psi`

with:

- `q_psi` produced by the existing trajectory branch
- `d_psi` produced by the existing current-state branch

For the first testable implementation:

- `L_transport = ||q_psi - q_boot||^2`
- `L_drift = ||d_psi - d_boot||^2`
- `L_id = ||(q_psi + d_psi) - m_id_existing||^2`

where `m_id_existing` remains the already-implemented Stage 05 aggregate corrected residual
identity target.

This first implementation choice is deliberate:

- it tests the v3-A working hypothesis with the smallest possible code change
- it does not yet introduce branchwise identity targets
- it does not yet introduce rollout-consistency or semigroup losses
- it does not yet claim that explicit transport-drift decomposition is a confirmed repair

#### 18.11.5 Non-goals and required next deliverables

Stage 05 v3-A must not be framed as:

- a reopening of Stage 04 package-internal work
- another pure same-family budget escalation
- another pure same-family efficiency tweak
- a default introduction of `muPC`, `iPC`, `DKP-PC`, or a full `AlphaFlow` family replacement
- a change to the artifact-independent target-construction boundary
- a promotion of task accuracy to the primary gate

The next Stage 05 v3-A implementation pass must minimally add:

- a new Stage 05 v3-A candidate codepath
- a new comparison entry or suite against the current Stage 05 v2 reference
- a matching smoke test
- a dedicated artifact directory
- aggregate summary fields that report:
  - whether explicit transport-drift decomposition is enabled
  - pairwise deltas versus the current v2 reference
  - a gap-closure style decision field
  - `recommended_next_move`

### 18.12 Stage 05 v3-B charter

This section opens the next Stage 05 charter and now also records the first minimal candidate
implementation contract.

It still does not redefine:

- the current Stage 05 transport family
- the explicit transport-drift decomposition from v3-A
- the Stage 05 mechanism-first acceptance framing

Current evidence motivating this charter:

- the fixed-budget `1536`-epoch `v2 vs v3-A` comparison shows positive gap closure versus the fixed-budget v2 reference
- the same comparison shows material configured-step mechanism improvement versus the fixed-budget v2 reference
- the same comparison does not show an obvious report-only accuracy regression under the current Stage 05 rule
- the next credible mechanism question is therefore no longer target-entanglement-only

#### 18.12.1 Working hypothesis

Stage 05 v3-B is motivated by the working hypothesis that, after explicit transport-drift
decomposition, the main remaining fixed-budget inefficiency is trajectory-level rather than
target-entanglement-level.

More specifically, the current residual MeanFlow family may still face an optimization conflict
between:

- direct horizon matching
- recursive continuation or trajectory consistency

This is a chartering hypothesis, not a proved repository conclusion.

#### 18.12.2 Trajectory curriculum split

Stage 05 v3-B keeps the current Stage 05 remaining-horizon notation:

- `t` is the current knot time
- `r` is the remaining horizon

Introduce a curriculum split parameter:

- `alpha in (0, 1]`

Define the intermediate knot time and remaining horizon after that knot by:

- `s = t + alpha * r`
- `r_s = (1 - alpha) * r`

The exact average velocity over the full interval can then be decomposed as:

- `u^*_{t,r} = alpha * u^*_{t, alpha r} + (1 - alpha) * u^*_{s, r_s}`

Interpretation:

- when `alpha = 1`, the contract reduces to direct full-interval trajectory matching
- when `alpha` becomes smaller, the contract shifts more mass onto recursive continuation over the remaining interval

This gives Stage 05 a natural curriculum axis without leaving the corrected residual MeanFlow
family.

#### 18.12.3 v3-B contract definition

Stage 05 v3-B is defined as a trajectory curriculum contract on top of the validated v3-A
explicit transport-drift contract.

It preserves:

- explicit transport-drift decomposition
- artifact-independent target construction
- mechanism-first evaluation
- task accuracy as a report-only signal
- the current fixed-budget shared comparison protocol

It changes only the higher-level trajectory supervision contract:

- how direct full-interval matching and recursive continuation are balanced during training

This charter does not yet fix:

- the exact curriculum schedule over `alpha`
- the exact v3-B loss form
- the exact weighting between direct and recursive trajectory terms
- the final rollout-consistency implementation details

Those choices can now be explored only inside narrow, explicitly documented candidate paths.

#### 18.12.4 Non-goals and required next deliverables

Stage 05 v3-B must not be framed as:

- a reopening of Stage 04 package-internal work
- another pure same-family budget escalation
- another pure same-family efficiency tweak
- a branch-gating variant
- a default introduction of `muPC`, `iPC`, `DKP-PC`, or a full `AlphaFlow` family replacement
- a change to the artifact-independent target-construction boundary
- a promotion of task accuracy to the primary gate

The next Stage 05 v3-B implementation pass must minimally add:

- a new Stage 05 v3-B candidate codepath on top of the current v3-A branch
- a new comparison entry or suite against:
  - the fixed-budget Stage 05 v2 control
  - the fixed-budget Stage 05 v3-A result
- a matching smoke test
- a dedicated artifact directory
- aggregate summary fields that report:
  - whether trajectory curriculum is enabled
  - the curriculum schedule identity
  - pairwise deltas versus the current v3-A result
  - pairwise deltas versus the current v2 control
  - a gap-closure style decision field
  - `recommended_next_move`

#### 18.12.5 First minimal v3-B candidate contract

The first implemented Stage 05 v3-B candidate is intentionally narrow.

It keeps the v3-A decomposition unchanged:

- `u_hat(z_t, r, t; c) = g_t + q_psi + d_psi`

It also keeps the current v3-A branchwise supervision unchanged:

- `L_transport`
- `L_drift`
- aggregate `L_id`

The new v3-B ingredient is one additional aggregate trajectory curriculum loss:

- `L_traj_curr`

For the first candidate, define:

- `u_boot_short = u_boot(z_t, alpha * r, t; c)`
- `z_s_boot = z_t + alpha * r * u_boot_short`
- `u_hat_cont = u_hat(z_s_boot, r_s, s; c)` evaluated on the target side only
- `u_curr_target = alpha * u_boot_short + (1 - alpha) * u_hat_cont`

The first candidate then uses:

- `L_traj_curr = || u_hat(z_t, r, t; c) - u_curr_target ||^2`

Equivalent residual-side form used by the implementation:

- `m_curr_target = u_curr_target - g_t`
- `L_traj_curr = || (q_psi + d_psi) - m_curr_target ||^2`

Normative first-pass implementation constraints:

- keep the continuation target aggregate, not branchwise
- keep the continuation evaluation detached on the target side
- do not add a new branchwise identity derivation in this pass
- do not add endpoint semigroup losses in this pass
- do not alter the v3-A explicit `q_boot / d_boot` decomposition in this pass

The first curriculum schedule is intentionally simple and inspectable:

- phase 1:
  - `alpha = 1`
  - `lambda_traj_curr = 0`
- phase 2:
  - `alpha` decreases smoothly toward a fixed `alpha_floor`
  - `lambda_traj_curr` ramps up smoothly
- phase 3:
  - `alpha = alpha_floor`
  - `lambda_traj_curr` stays fixed

The first implemented schedule must keep:

- `0 < alpha_floor < 1`
- no collapse to `alpha = 0`

This first v3-B candidate is still a working-hypothesis test, not a proved superior family.

### 18.13 Stage 05 v3-C charter

This section opens the next Stage 05 charter only.

It does not itself implement:

- a new Stage 05 loss term
- a new Stage 05 experiment suite
- a new Stage 05 default or replacement claim

#### 18.13.1 Current post-promotion motivation

Current evidence after the refined v3-B promotion is:

- the fixed-budget `v2 vs v3-A` result already shows that explicit transport-drift decomposition materially improves configured-step mechanism over the fixed-budget v2 control
- the refined fixed-budget v3-B recompare then shows that `stage05_v3b_stronger_traj_curr_weight` materially improves configured-step mechanism again over the fixed-budget v3-A reference
- the current Stage 05 line therefore already has evidence that:
  - target entanglement matters
  - trajectory-level supervision matters
- the remaining fixed-budget gap is now more plausibly tied to a still-missing higher-level consistency contract than to another immediate branch-local or weight-local tweak

Current ranked post-promotion mechanism suspects are:

- endpoint / semigroup consistency contract
- still-insufficient trajectory curriculum contract
- corrected residual identity contract

Stage 05 v3-C is opened because the first item is now the most credible next mechanism question.

#### 18.13.2 Working hypothesis

Stage 05 v3-C is motivated by the working hypothesis that, after:

- explicit transport-drift decomposition in v3-A
- promoted refined trajectory curriculum in v3-B

the current Stage 05 family still lacks an explicit endpoint / semigroup consistency contract
across split horizons, and that this missing contract is now a more credible configured-step
limiter than another immediate trajectory-weight or identity-only refinement.

This is a chartering hypothesis, not a proved repository conclusion.

#### 18.13.3 Endpoint / semigroup consistency notation

Stage 05 v3-C keeps the current Stage 05 remaining-horizon notation:

- `t` is the current knot time
- `r` is the remaining horizon
- `alpha in (0, 1]`
- `s = t + alpha * r`
- `r_s = (1 - alpha) * r`

Define the direct full-horizon endpoint prediction by:

- `z_hat_direct = z_t + r * u_hat(z_t, r, t; c)`

Define the split-horizon midpoint prediction by:

- `z_hat_mid = z_t + alpha * r * u_hat(z_t, alpha * r, t; c)`

Define the split-horizon endpoint prediction by:

- `z_hat_split = z_t + alpha * r * u_hat(z_t, alpha * r, t; c) + (1 - alpha) * r * u_hat(z_hat_mid, r_s, s; c)`

Define the semigroup consistency residual by:

- `Delta_sg = z_hat_direct - z_hat_split`

#### 18.13.4 v3-C contract definition

Stage 05 v3-C is defined as an endpoint / semigroup consistency contract on top of the promoted
refined v3-B scaffold.

It preserves:

- explicit transport-drift decomposition from v3-A
- the promoted refined v3-B trajectory-curriculum scaffold
- artifact-independent target construction
- mechanism-first evaluation
- task accuracy as a report-only signal

It changes only the next higher-level consistency question:

- whether direct full-horizon endpoint prediction and split-horizon endpoint prediction should be constrained more explicitly across the same interval family

This charter does not yet fix:

- the exact semigroup-consistency loss form
- the exact weighting between existing trajectory-curriculum terms and any new semigroup term
- whether the split endpoint should be enforced on velocity, endpoint state, or both
- the exact schedule over `alpha`

Those choices belong to the next implementation pass.

#### 18.13.5 Why endpoint / semigroup consistency now ranks first

Current repository evidence does not rank corrected residual identity as the lead next issue,
because:

- the current Stage 05 scaffold already uses an explicit corrected residual identity target
- the current v2 and later scaffolds already keep anchor, trajectory, and state derivative terms explicit

Current repository evidence also does not rank another immediate v3-B-only trajectory tweak as
the lead next issue, because:

- refined v3-B already became the active fixed-budget improvement reference
- the remaining gap after that promotion is better described as a missing higher-level consistency layer than as an unanswered promotion question

The current v3-B scaffold still lacks an explicit endpoint / semigroup consistency contract:

- it adds trajectory curriculum
- it still does not directly constrain `Delta_sg`

This makes endpoint / semigroup consistency the most credible next charter from the current state.

#### 18.13.6 Non-goals and required next deliverables

Stage 05 v3-C must not be framed as:

- a reopening of Stage 04 package-internal work
- another pure same-family budget escalation
- another pure same-family efficiency tweak
- a broad v3-B parameter search
- a default introduction of `muPC`, `iPC`, `DKP-PC`, or a full `AlphaFlow` family replacement
- a promotion of task accuracy to the primary gate
- a replacement claim against the frozen Stage 04 bridge result on `main`

The next Stage 05 v3-C implementation pass must minimally add:

- a new Stage 05 v3-C candidate codepath on top of `stage05_v3b_stronger_traj_curr_weight`
- a new comparison entry or suite against:
  - the fixed-budget Stage 05 v2 control
  - the promoted refined Stage 05 v3-B reference
- a matching smoke test
- a dedicated artifact directory
- aggregate summary fields that report:
  - whether endpoint / semigroup consistency is enabled
  - the split-horizon semigroup identity
  - pairwise deltas versus the promoted refined v3-B reference
  - pairwise deltas versus the fixed-budget v2 control
  - a gap-closure style decision field
  - `recommended_next_move`

#### 18.13.7 First diagnostic-only v3-C probe contract

The first implemented Stage 05 v3-C pass must remain diagnostic-only.

It does not yet claim that endpoint / semigroup consistency is the final main Stage 05 contract.
It only tests whether an explicit semigroup probe adds useful mechanism signal on top of the
promoted refined v3-B scaffold.

The first implemented v3-C candidate must therefore:

- preserve the promoted refined v3-B trajectory curriculum scaffold
- preserve the explicit v3-A transport-drift decomposition
- preserve artifact-independent target construction
- keep the semigroup probe aggregate-only in the first pass

For the first pass, define:

- prediction side:
  - `z_hat_direct = z_t + r * u_hat(z_t, r, t; c)`
- target side:
  - `z_hat_split_target = stopgrad(z_hat_split)`
  - where `z_hat_split` is the split-horizon endpoint prediction defined above

The first-pass semigroup probe loss is then:

- `L_sg = || z_hat_direct - z_hat_split_target ||^2`

The first-pass total objective is allowed to stay minimal:

- `L_total_v3c = L_promoted_v3b + lambda_sg * L_sg`

where:

- `L_promoted_v3b` is the already-implemented promoted refined v3-B objective
- `lambda_sg` is explicit, small, and reviewable

The first implementation pass may realize this endpoint loss through an exactly equivalent
residual-target proxy, provided that:

- the split endpoint remains single-sided detached on the target side
- the proxy remains mathematically consistent with the endpoint objective
- the implementation keeps the semigroup target mode explicit in artifacts

The first-pass v3-C implementation must not yet add:

- branchwise semigroup targets
- multiple semigroup variants
- a broad semigroup schedule search
- a claim that v3-C should replace the promoted refined v3-B reference before a real fixed-budget comparison

#### 18.13.8 Post-promotion structural interpretation

Current repository evidence now covers the following chain:

- `v2 -> v3-A`
  - explicit transport-drift decomposition materially improved configured-step mechanism over the fixed-budget v2 control
- `v3-A -> refined v3-B`
  - stronger trajectory curriculum materially improved configured-step mechanism again over the fixed-budget v3-A reference
- `refined v3-B -> refined v3-C`
  - explicit endpoint / semigroup consistency then materially improved configured-step mechanism again under the formal fixed-budget recompare

The repository therefore now adopts the structural interpretation:

- `absorb_semigroup_into_main_trajectory_contract`

Interpretation:

- endpoint / semigroup consistency no longer looks like a permanently attached auxiliary-only probe
- it currently looks complementary to the main trajectory contract and should be internalized into that contract in future mainline work
- current evidence still does not justify `refactor_main_contract_around_endpoint_semigroup_consistency`
- trajectory curriculum therefore remains part of the main Stage 05 contract framing

This remains a repository-level working interpretation, not a proved mathematical theorem.

No new Stage 05 charter is opened by this conclusion.

The next implementation question inside the existing Stage 05 family is:

- how to refactor the current refined v3-C layering into one main trajectory contract that internalizes semigroup consistency without continuing additive loss stacking

#### 18.13.9 First exact-fusion consolidation pass

The first contract-consolidation implementation inside the current Stage 05 family is:

- `stage05_v3c_fused_trajectory_semigroup_contract`

It keeps:

- the Stage 05 v3-A explicit transport-drift decomposition
- the Stage 05 v3-B trajectory-curriculum scaffold
- the Stage 05 v3-C single-sided detached semigroup target construction

It then replaces the stacked trajectory-level update:

- `lambda_tc * ||m_hat - m_traj_star||^2`
- `+ lambda_sg * r^2 * ||m_hat - m_sg_star||^2`

with one unified detached-target main trajectory contract:

- `W = lambda_tc + lambda_sg * r^2`
- `m_fuse_star = (lambda_tc * m_traj_star + lambda_sg * r^2 * m_sg_star) / W`
- `L_main_traj = W * ||m_hat - m_fuse_star||^2`

Interpretation:

- this is an exact detached-target fusion move inside the current Stage 05 implementation style
- it is a contract-consolidation step, not a new mechanism family
- it absorbs semigroup consistency into the main trajectory contract rather than leaving it as an attached auxiliary-only update
- it still does not justify `refactor_main_contract_around_endpoint_semigroup_consistency`

Current evidence from the first fixed-budget fused-contract comparison is:

- the fused candidate keeps one-step and configured-step mechanism positive
- it avoids an obvious report-only accuracy regression relative to `stage05_v3c_stronger_semigroup_weight`
- it shows a positive but very small contextual gap-closure movement beyond the active refined v3-C reference
- it does not yet materially improve configured-step mechanism over `stage05_v3c_stronger_semigroup_weight`

The next implementation question therefore becomes:

- how to refine the fusion direction so the consolidated main trajectory contract can materially beat the active refined v3-C reference under the existing mechanism-first rule

#### 18.13.10 First non-equivalent midpoint-reconstruction pass

The next contract-consolidation implementation inside the current Stage 05 family is:

- `stage05_v3c_midpoint_reconstructed_trajectory_contract`

This pass is opened because the first exact detached-target fusion pass was semantically clean but
did not materially beat the active refined `stage05_v3c_stronger_semigroup_weight` reference.

The next move therefore changes the target-construction rule itself instead of re-expressing the
same stacked detached targets.

It keeps:

- the Stage 05 v3-A explicit transport-drift decomposition
- the Stage 05 v3-B trajectory-curriculum split notation
- the Stage 05 v3-C single-sided detached semigroup endpoint target

It then reconstructs the internal knot before continuation is re-evaluated.

Using the current notation:

- `u_B = u_boot(z_t, alpha * r, t; c)`
- `z_B = z_t + alpha * r * u_B`
- `u_C^B = stopgrad(u_hat(z_B, r_s, s; c))`
- `z_sg^* = stopgrad(z_hat_split)`
- `kappa = (lambda_sg * r^2) / (lambda_tc + lambda_sg * r^2)`
- `z_sg_mid^* = z_sg^* - (1 - alpha) * r * u_C^B`
- `z_mid^* = (1 - kappa) * z_B + kappa * z_sg_mid^*`
- `u_short^* = (z_mid^* - z_t) / (alpha * r)`
- `u_C^* = stopgrad(u_hat(z_mid^*, r_s, s; c))`
- `u_main^* = alpha * u_short^* + (1 - alpha) * u_C^*`
- `m_main^* = u_main^* - g_t`

The unified main trajectory contract is then:

- `lambda_main = lambda_tc + lambda_sg * r^2`
- `L_main_traj = lambda_main * ||m_hat - m_main^*||^2`

Interpretation:

- semigroup consistency is absorbed into the main trajectory contract through midpoint reconstruction
- continuation is re-evaluated at the reconstructed midpoint, so this pass is not gradient-equivalent to the previous exact fused detached-target contract
- the candidate remains inside the current Stage 05 family and does not refactor the family around semigroup consistency alone

This pass still does not justify:

- `refactor_main_contract_around_endpoint_semigroup_consistency`
- a reopening of Stage 04
- a new mechanism family

#### 18.13.11 Endpoint-line midpoint refinement

The next narrow refinement inside the same consolidation direction is:

- `stage05_v3c_endpoint_line_midpoint_trajectory_contract`

This pass is opened because the first midpoint-reconstructed candidate:

- stayed directionally correct
- but did not materially beat the active refined `stage05_v3c_stronger_semigroup_weight`
- and its midpoint guidance still backed semigroup endpoint information out through the bootstrap continuation

The concrete refinement is:

- keep the bootstrap short leg
- keep detached continuation re-evaluation at the reconstructed midpoint
- replace continuation-backout midpoint guidance with endpoint-line midpoint guidance

Using the current notation:

- `u_B = u_boot(z_t, alpha * r, t; c)`
- `z_B = z_t + alpha * r * u_B`
- `u_C^B = stopgrad(u_hat(z_B, r_s, s; c))`
- `z_sg^* = stopgrad(z_hat_split)`
- `u_sg^* = (z_sg^* - z_t) / r`
- `kappa = (lambda_sg * r^2) / (lambda_tc + lambda_sg * r^2)`
- `z_line_mid^* = z_t + alpha * (z_sg^* - z_t)`
- `z_mid^* = (1 - kappa) * z_B + kappa * z_line_mid^*`
- `u_short^* = (z_mid^* - z_t) / (alpha * r)`
- `u_C^* = stopgrad(u_hat(z_mid^*, r_s, s; c))`
- `u_main^* = alpha * u_short^* + (1 - alpha) * u_C^*`
- `m_main^* = u_main^* - g_t`

The unified main trajectory contract remains:

- `lambda_main = lambda_tc + lambda_sg * r^2`
- `L_main_traj = lambda_main * ||m_hat - m_main^*||^2`

Interpretation:

- semigroup consistency remains absorbed into the main trajectory contract
- trajectory remains the main contract frame
- the refinement is more non-equivalent than exact detached-target fusion because continuation is re-evaluated after the midpoint is changed in state space
- the refinement still does not justify refactoring the family around endpoint / semigroup consistency alone

Current fixed-budget result:

- the endpoint-line midpoint candidate improves configured-step mechanism and contextual gap closure beyond `stage05_v3c_stronger_semigroup_weight`
- it avoids an obvious report-only accuracy regression
- it does not materially beat `stage05_v3c_stronger_semigroup_weight` under the current threshold

Current decision:

- keep `stage05_v3c_stronger_semigroup_weight` as the active fixed-budget Stage 05 improvement reference
- keep the endpoint-line midpoint direction alive as the current refinement direction
- do not treat this result as a promotion or as a reason to reopen Stage 04

#### 18.13.12 Endpoint-line continuation-target refinement

The next narrow refinement inside the same consolidation direction is:

- `stage05_v3c_endpoint_line_continuation_blend_trajectory_contract`

This pass is opened because the endpoint-line midpoint candidate:

- already showed directional configured-step improvement over the active refined `stage05_v3c_stronger_semigroup_weight`
- already avoided an obvious report-only accuracy regression
- but still left the continuation leg too close to a pure re-evaluated trajectory continuation

The concrete refinement is therefore:

- keep the endpoint-line midpoint reconstruction
- keep trajectory as the main contract frame
- inject semigroup geometry directly into the continuation target
- do not reintroduce a separate auxiliary semigroup loss

Using the current notation:

- `u_B = u_boot(z_t, alpha * r, t; c)`
- `z_B = z_t + alpha * r * u_B`
- `u_C^B = stopgrad(u_hat(z_B, r_s, s; c))`
- `z_sg^* = stopgrad(z_hat_split)`
- `u_sg^* = (z_sg^* - z_t) / r`
- `kappa = (lambda_sg * r^2) / (lambda_tc + lambda_sg * r^2)`
- `z_line_mid^* = z_t + alpha * (z_sg^* - z_t)`
- `z_mid^* = (1 - kappa) * z_B + kappa * z_line_mid^*`
- `u_short^* = (z_mid^* - z_t) / (alpha * r)`
- `u_C,traj^* = stopgrad(u_hat(z_mid^*, r_s, s; c))`
- `u_C,sg^* = (z_sg^* - z_mid^*) / ((1 - alpha) * r)`
- `u_C^* = (1 - kappa) * u_C,traj^* + kappa * u_C,sg^*`
- `u_main^* = alpha * u_short^* + (1 - alpha) * u_C^*`
- `m_main^* = u_main^* - g_t`

The unified main trajectory contract remains:

- `lambda_main = lambda_tc + lambda_sg * r^2`
- `L_main_traj = lambda_main * ||m_hat - m_main^*||^2`

Interpretation:

- semigroup consistency remains absorbed into the main trajectory contract
- the continuation target is now explicitly refined, not only the midpoint state
- trajectory remains the main contract frame
- the candidate still does not justify `refactor_main_contract_around_endpoint_semigroup_consistency`

This pass must not be implemented as:

- another exact detached-target folding pass
- another weight-only tweak
- a return to `L_traj_curr + lambda_sg * L_sg` as separate trajectory-level terms

Current fixed-budget result:

- the endpoint-line continuation-blend candidate improves configured-step mechanism and contextual gap closure beyond `stage05_v3c_stronger_semigroup_weight`
- it avoids an obvious report-only accuracy regression
- it does not materially beat `stage05_v3c_stronger_semigroup_weight` under the current threshold

Current decision:

- keep `stage05_v3c_stronger_semigroup_weight` as the active fixed-budget Stage 05 improvement reference
- keep `stage05_v3c_endpoint_line_continuation_blend_trajectory_contract` as the current narrow refinement candidate
- keep the continuation-target refinement direction alive, but refine implementation rather than promote the current candidate

#### 18.13.13 Coupled local defect-projection refinement

The next narrow refinement inside the same consolidation direction is:

- `stage05_v3c_coupled_defect_projection_trajectory_contract`

This pass is opened because the endpoint-line continuation-blend candidate:

- improved configured-step mechanism beyond `stage05_v3c_stronger_semigroup_weight`
- avoided an obvious report-only accuracy regression
- but still behaved like a sequential one-pass midpoint update followed by a continuation update

The concrete refinement is therefore:

- keep the endpoint-line midpoint predictor
- keep semigroup consistency absorbed into the main trajectory contract
- replace the sequential midpoint / continuation refinement with one shared local semigroup-defect projection
- allow one detached continuation corrector re-evaluation only
- do not introduce a new free coupling hyperparameter
- do not reintroduce a separate auxiliary semigroup loss

Using the current notation:

- `u_B = u_boot(z_t, alpha * r, t; c)`
- `z_B = z_t + alpha * r * u_B`
- `z_sg^* = stopgrad(z_hat_split)`
- `u_sg^* = (z_sg^* - z_t) / r`
- `kappa = (lambda_sg * r^2) / (lambda_tc + lambda_sg * r^2)`
- `z_line_mid^* = z_t + alpha * (z_sg^* - z_t)`
- `z_mid^(0) = (1 - kappa) * z_B + kappa * z_line_mid^*`
- `u_short^(0) = (z_mid^(0) - z_t) / (alpha * r)`
- `u_C,traj^(0) = stopgrad(u_hat(z_mid^(0), r_s, s; c))`
- `d_sg^(0) = u_sg^* - (alpha * u_short^(0) + (1 - alpha) * u_C,traj^(0))`
- `rho = (lambda_sg * r^2) / (lambda_tc + lambda_sg * r^2 * (alpha^2 + (1 - alpha)^2))`
- `u_short^(1/2) = u_short^(0) + rho * alpha * d_sg^(0)`
- `u_C^(1/2) = u_C,traj^(0) + rho * (1 - alpha) * d_sg^(0)`
- `z_mid^(1/2) = z_t + alpha * r * u_short^(1/2)`
- `u_C,traj^(1) = stopgrad(u_hat(z_mid^(1/2), r_s, s; c))`
- `d_sg^(1) = u_sg^* - (alpha * u_short^(1/2) + (1 - alpha) * u_C,traj^(1))`
- `u_short^* = u_short^(1/2) + rho * alpha * d_sg^(1)`
- `u_C^* = u_C,traj^(1) + rho * (1 - alpha) * d_sg^(1)`
- `u_main^* = alpha * u_short^* + (1 - alpha) * u_C^*`
- `m_main^* = u_main^* - g_t`

The unified main trajectory contract remains:

- `lambda_main = lambda_tc + lambda_sg * r^2`
- `L_main_traj = lambda_main * ||m_hat - m_main^*||^2`

Normative implementation constraints:

- keep `z_sg^*` single-sided detached for the whole local defect-projection pass
- do not divide by `(1 - alpha)` anywhere in this candidate
- do not add a second extra corrector beyond the single detached continuation re-evaluation above
- expose explicit artifact fields for:
  - coupled defect projection enabled
  - shared semigroup defect coupling enabled
  - predictor-corrector refinement enabled
  - second-pass continuation re-evaluation enabled
  - defect-projection coefficient identity

Interpretation:

- semigroup consistency remains absorbed into the main trajectory contract
- trajectory remains the main contract frame
- midpoint and continuation are now corrected as one local two-segment defect problem instead of a sequential tweak
- this still does not justify `refactor_main_contract_around_endpoint_semigroup_consistency`

#### 18.13.14 Precision-weighted continuation-corrector refinement

The next narrow asymmetric refinement tested inside the same consolidation direction is:

- `stage05_v3c_precision_weighted_continuation_corrector_trajectory_contract`

This pass keeps the endpoint-line midpoint predictor fixed and refines only the continuation-side coefficient.

Using the current notation:

- `u_B = u_boot(z_t, alpha * r, t; c)`
- `z_B = z_t + alpha * r * u_B`
- `z_sg^* = stopgrad(z_hat_split)`
- `u_sg^* = (z_sg^* - z_t) / r`
- `kappa = (lambda_sg * r^2) / (lambda_tc + lambda_sg * r^2)`
- `z_line_mid^* = z_t + alpha * (z_sg^* - z_t)`
- `z_mid^* = (1 - kappa) * z_B + kappa * z_line_mid^*`
- `u_short^* = (z_mid^* - z_t) / (alpha * r)`
- `u_C,traj^* = stopgrad(u_hat(z_mid^*, r_s, s; c))`
- `eta_cont = (lambda_sg * r^2 * (1 - alpha)^2) / (lambda_tc + lambda_sg * r^2 * (1 - alpha)^2)`
- `u_C,sg^* = (z_sg^* - z_mid^*) / ((1 - alpha) * r)` for active continuation segments only
- `u_C^* = (1 - eta_cont) * u_C,traj^* + eta_cont * u_C,sg^*`
- equivalently at full-horizon level:
  - `u_main^* = (1 - eta_cont) * (alpha * u_short^* + (1 - alpha) * u_C,traj^*) + eta_cont * u_sg^*`
- `m_main^* = u_main^* - g_t`

The unified main trajectory contract remains:

- `lambda_main = lambda_tc + lambda_sg * r^2`
- `L_main_traj = lambda_main * ||m_hat - m_main^*||^2`

Normative implementation constraints:

- keep the endpoint-line midpoint predictor unchanged from `stage05_v3c_endpoint_line_midpoint_trajectory_contract`
- keep only one detached continuation re-evaluation at the reconstructed midpoint
- do not add a second midpoint feedback step
- do not add a second continuation re-evaluation
- do not reintroduce a separate auxiliary semigroup loss
- expose explicit artifact fields for:
  - precision-weighted continuation corrector enabled
  - continuation-MAP closed-form coefficient enabled
  - continuation coefficient identity

Current fixed-budget result:

- the precision-weighted continuation-corrector candidate directionally improves configured-step mechanism over `stage05_v3c_stronger_semigroup_weight`
- it improves contextual gap closure beyond the active refined v3-C reference
- it avoids an obvious report-only accuracy regression
- it does not materially beat `stage05_v3c_stronger_semigroup_weight` under the current threshold
- its configured-step gain is weaker than the earlier `stage05_v3c_endpoint_line_continuation_blend_trajectory_contract` result

Current decision:

- keep `stage05_v3c_stronger_semigroup_weight` as the active fixed-budget Stage 05 improvement reference
- treat `stage05_v3c_precision_weighted_continuation_corrector_trajectory_contract` as a tested asymmetric continuation-corrector variant rather than the active reference
- keep the continuation-target refinement direction alive, but do not treat this exact `eta_cont` weighting as the new same-family leader

#### 18.13.15 Final single-axis continuation-strength diagnostic

The final narrow diagnostic inside the current v3-C contract-consolidation micro-family is:

- `stage05_v3c_scaled_continuation_blend_trajectory_contract`

This pass does not open a new Stage 05 family. It keeps the exact endpoint-line continuation-blend scaffold and changes only the continuation-side semigroup correction strength.

Normative constraints:

- keep the endpoint-line midpoint reconstruction unchanged
- keep exactly one detached continuation re-evaluation at the reconstructed midpoint
- do not add a second midpoint feedback step
- do not add a second continuation re-evaluation
- do not enable coupled defect projection
- do not reintroduce a separate auxiliary semigroup loss
- do not sweep a scale zoo inside this pass

Using the current notation:

- `u_B = u_boot(z_t, alpha * r, t; c)`
- `z_B = z_t + alpha * r * u_B`
- `z_sg^* = stopgrad(z_hat_split)`
- `u_sg^* = (z_sg^* - z_t) / r`
- `kappa = (lambda_sg * r^2) / (lambda_tc + lambda_sg * r^2)`
- `z_line_mid^* = z_t + alpha * (z_sg^* - z_t)`
- `z_mid^* = (1 - kappa) * z_B + kappa * z_line_mid^*`
- `u_short^* = (z_mid^* - z_t) / (alpha * r)`
- `u_C,traj^* = stopgrad(u_hat(z_mid^*, r_s, s; c))`
- `u_C,sg^* = (z_sg^* - z_mid^*) / ((1 - alpha) * r)` for active continuation segments only
- `gamma_cont = 1.5`
- `kappa_eff = min(1.0, gamma_cont * kappa)`
- `u_C^* = (1 - kappa_eff) * u_C,traj^* + kappa_eff * u_C,sg^*`
- `u_main^* = alpha * u_short^* + (1 - alpha) * u_C^*`
- `m_main^* = u_main^* - g_t`

The unified main trajectory contract remains:

- `lambda_main = lambda_tc + lambda_sg * r^2`
- `L_main_traj = lambda_main * ||m_hat - m_main^*||^2`

Interpretation:

- semigroup consistency remains absorbed into the main trajectory contract
- trajectory remains the main contract frame
- this is a final single-axis strength diagnostic inside the existing micro-family, not a new mechanism family
- this pass still does not justify `refactor_main_contract_around_endpoint_semigroup_consistency`
