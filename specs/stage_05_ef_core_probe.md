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

This section opens the next Stage 05 charter only.

It does not itself modify:

- the current Stage 05 v3-A implementation
- the current Stage 05 experiment entrypoints
- the current Stage 05 smoke or fixed-budget comparison code

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

Those choices belong to the next implementation pass.

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
