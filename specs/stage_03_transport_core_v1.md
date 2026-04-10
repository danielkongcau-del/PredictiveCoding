# FMPC Stage 03 Transport Core v1 Addendum

This file extends the baseline math in [spec_math.md](/e:/CodeSpace/PredictiveCoding/spec_math.md).

To build a complete mathematical understanding for Stage 03 or later, read in this order:

1. [spec_math.md](/e:/CodeSpace/PredictiveCoding/spec_math.md)
2. this addendum
3. a later stage addendum only if the task explicitly touches that later stage

Scope and precedence:

- this addendum overrides the baseline only inside the explicit Stage 03 transport scope
- outside that scope, the baseline remains authoritative
- this addendum does not silently redefine the baseline energy, baseline hidden-state gradient, or baseline local parameter-update equations

## 16. Transport Core v1 addendum

This addendum defines the first teacher-free FMPC transport contract without changing
the baseline predictive-coding energy, hidden-state gradient, or local parameter-update
mathematics.

### 16.1 Scope

Transport Core v1 applies only to the existing layered predictive-coding substrate.

- it does not define a general PCG substrate
- it does not redefine predict-mode inference
- it does not replace the baseline iterative PC algorithm outside the explicit
  teacher-free transport experiment path

### 16.2 Training context and hidden latent

Transport Core v1 uses the supervised training context:

- `c = (x, y)`
- `x^0 = x` remains clamped
- `x^L = y` remains clamped

The flattened hidden latent is:

- `z = flatten(x^1, ..., x^(L-1))`

This uses the repository's existing hidden-state flattening contract:

- `x^0` is never included in `z`
- `x^L` is never included in `z`
- only free hidden layers `x^1 .. x^(L-1)` are concatenated

### 16.3 Local energy substrate and instantaneous flow

Transport Core v1 reuses the baseline predictive-coding energy:

- `E_theta(z; c) := F(states(z; x, y), theta)`

where:

- `states(z; x, y)` reconstructs the full state list from the flattened hidden state
  and the clamped training context
- `F` is exactly the baseline squared-error predictive-coding energy already defined
  in this spec

The instantaneous hidden-state flow is:

- `g_theta(z; c) = -∇_z E_theta(z; c)`

No teacher approximation is assumed in this definition.

### 16.4 Average-velocity model and time contract

Transport Core v1 introduces an average-velocity model:

- `u_psi(z_t, r, t; c)`

with:

- `t = current time`
- `r = remaining horizon`
- `0 <= t < 1`
- `0 < r <= 1 - t`

At rollout knot `k`, we write:

- `t_k` for the current time
- `r_k = 1 - t_k` for the remaining horizon
- `Δt_k = t_{k+1} - t_k`

`u_psi(z_k, r_k, t_k; c) is interpreted as an estimate of the average velocity over the remaining horizon [t_k, 1], and the update z_{k+1} = z_k + Δt_k * u_psi(...) is a piecewise-constant coarse transport approximation, not a redefinition of the instantaneous flow.`

The coarse transport update is:

- `z_{k+1} = z_k + Δt_k * u_psi(z_k, r_k, t_k; c)`

### 16.5 Fixed-terminal-time MeanFlow identity direction

Transport Core v1 uses the fixed-terminal-time direction:

- `(dt, dr) = (+1, -1)`

This means the total derivative is taken along trajectories that keep:

- `t + r = const`

equivalently, along a direction that advances the current time while preserving the
same terminal time.

The MeanFlow-style identity target is therefore based on:

- `u(z_t, r, t; c) ≈ g_theta(z_t; c) + r * D_T u(z_t, r, t; c)`

where `D_T` denotes the total derivative in the fixed-terminal-time direction.

### 16.5.1 Feature-dependent psi inputs and truncated identity semantics

When `u_psi` consumes appended current-state teacher-free features such as:

- `g_t`
- `e_out_t`
- `F_t`

the full fixed-terminal-time identity requires chain-rule directional-derivative
terms through those appended feature blocks.

The repository therefore distinguishes two identity-tangent semantics:

- `feature_aware_tangents = true`
  - inject finite-difference directional derivatives of the appended feature block
    along `g_t`
  - this is the repository's explicit approximation to the full total derivative of
    the augmented input contract
- `feature_aware_tangents = false`
  - treat the appended feature block as frozen side information inside the JVP path
  - the resulting identity target is an explicit **truncated** fixed-terminal-time
    identity approximation, not the full total derivative through the augmented
    feature-dependent input

Current Incremental Bridge contract note:

- the canonical Incremental Bridge default may keep the truncated feature-frozen identity
  approximation when matched validation runs do not show a stable empirical gain from
  feature-aware tangents

### 16.6 Parameter updates after transport

Transport Core v1 changes only the hidden-state transport path used during training.

After transport produces a terminal hidden state `z_hat`, the repository still applies
the same local parameter-update rule already defined for the baseline:

1. reconstruct the full state list from `z_hat` and the clamped training context
2. recompute cache terms
3. compute baseline local parameter gradients
4. apply the same explicit parameter descent update

This addendum therefore does **not** redefine:

- the baseline energy
- the baseline hidden-state gradient
- the baseline local parameter-update equations


