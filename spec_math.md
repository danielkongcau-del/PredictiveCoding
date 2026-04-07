# Mathematical Specification

This file defines the **baseline mathematical formulation** for the repository.

It is intentionally narrow. The goal is to make implementation, testing, and later extension unambiguous.

---

## 1. Scope of this baseline

This baseline implements a **supervised, fully-connected, batch-first predictive coding network** with:

- layerwise deterministic activations
- squared-error energy
- iterative state inference for free layers
- local parameter updates
- batch-first arrays

This is **not** claimed to be the only or most general predictive coding formulation. It is the starting point for a clean implementation.

---

## 2. Notation

Let the network have layers indexed by `l = 0, 1, ..., L`.

- `X^l in R^(B x n_l)` is the state at layer `l`
- `B` is batch size
- `n_l` is the feature dimension of layer `l`
- `W^l in R^(n_l x n_(l-1))` maps layer `l-1 -> l`
- `b^l in R^(n_l,)` is the bias for layer `l`
- `A^l in R^(B x n_l)` is the pre-activation for layer `l`
- `MU^l in R^(B x n_l)` is the prediction of `X^l`
- `E^l in R^(B x n_l)` is the prediction error at layer `l`
- `phi_l(.)` is the activation for layer `l`
- `phi_l'` is its elementwise derivative
- `sigma_l^2 > 0` is the error variance scale for layer `l`

### Batch-first convention

All public-facing arrays use shape:

```text
(batch, features)
```

We use row-major sample organization.

---

## 3. Forward prediction equations

For each layer `l = 1, ..., L`:

```text
A^l = X^(l-1) @ (W^l)^T + b^l
MU^l = phi_l(A^l)
E^l = X^l - MU^l
```

Interpretation:

- the state at layer `l` is compared against the prediction made from layer `l-1`
- `E^l` is the local mismatch between the current state and its prediction

---

## 4. Energy / objective

For a batch, define the baseline energy as:

```text
F(X, W, b) = sum_{l=1}^L [ 1 / (2 B sigma_l^2) ] * || E^l ||_F^2
```

where `|| . ||_F` is the Frobenius norm.

Equivalent expanded form:

```text
F = sum_{l=1}^L [ 1 / (2 B sigma_l^2) ] * sum_{i=1}^B sum_{j=1}^{n_l} (E^l_{ij})^2
```

### Default simplification

In Phase 0, it is acceptable to set:

```text
sigma_l^2 = 1 for all l
```

but the implementation should keep the scaling explicit in the code structure.

---

## 5. Clamping rules

### Training mode

During supervised training:

- `X^0` is clamped to the input batch
- `X^L` is clamped to the target batch
- hidden layers `X^1 ... X^(L-1)` are free variables inferred by iterative dynamics

### Prediction / evaluation mode

During prediction:

- `X^0` is clamped to the input batch
- `X^1 ... X^L` are free variables, unless a specific experiment chooses to clamp additional layers
- the final prediction is read from `X^L` after inference

In early classification experiments, predicted class is:

```text
argmax(X^L, axis=1)
```

---

## 6. State initialization

State initialization is not part of the energy itself, but it matters numerically.

The baseline implementation should support at least these initialization modes for free layers:

1. `zeros`
   - initialize all free states to zero

2. `forward`
   - recursively initialize
   - `X^1 = phi_1(X^0 @ (W^1)^T + b^1)`
   - `X^2 = phi_2(X^1 @ (W^2)^T + b^2)`
   - etc.

Default recommendation for early experiments:

- use `forward` initialization when possible

---

## 7. Inference dynamics for free states

For a free hidden layer `l` with `1 <= l <= L-1`, update by gradient descent on `F` with respect to `X^l`.

The gradient is:

```text
dF/dX^l = (1 / (B sigma_l^2)) E^l
          - (1 / (B sigma_(l+1)^2)) (E^(l+1) ⊙ phi_(l+1)'(A^(l+1))) @ W^(l+1)
```

where `⊙` is elementwise multiplication.

The baseline explicit Euler update is:

```text
X^l <- X^l - eta_x * dF/dX^l
```

or equivalently:

```text
X^l <- X^l + eta_x * [
    -(1 / (B sigma_l^2)) E^l
    + (1 / (B sigma_(l+1)^2)) (E^(l+1) ⊙ phi_(l+1)'(A^(l+1))) @ W^(l+1)
]
```

### Output layer in prediction mode

If `X^L` is free during prediction, then:

```text
dF/dX^L = (1 / (B sigma_L^2)) E^L
```

and the update is:

```text
X^L <- X^L - eta_x * dF/dX^L
```

### Clamped layers

For any clamped layer, **do not update the state**.

### Inference schedule

A baseline inference pass runs for `T` steps:

```text
for t in 1..T:
    recompute A^l, MU^l, E^l for all l
    update each free layer according to the state rule
```

Synchronous versus in-place layer updates should be documented. The baseline recommendation is:

- recompute all errors from current states
- compute all state deltas
- apply the deltas layerwise after the full sweep

This avoids accidental order dependence.

---

## 8. Parameter updates

After inference has finished for the current batch, update parameters by descending `F` with respect to `W^l` and `b^l`.

For each layer `l = 1, ..., L`:

```text
dF/dW^l = -(1 / (B sigma_l^2)) (E^l ⊙ phi_l'(A^l))^T @ X^(l-1)
```

```text
dF/db^l = -(1 / (B sigma_l^2)) sum_rows(E^l ⊙ phi_l'(A^l))
```

where `sum_rows(.)` sums across the batch dimension and returns shape `(n_l,)`.

The baseline SGD-style updates are:

```text
W^l <- W^l - eta_w * dF/dW^l
b^l <- b^l - eta_b * dF/db^l
```

Equivalent additive form:

```text
W^l <- W^l + eta_w * (1 / (B sigma_l^2)) (E^l ⊙ phi_l'(A^l))^T @ X^(l-1)
```

```text
b^l <- b^l + eta_b * (1 / (B sigma_l^2)) sum_rows(E^l ⊙ phi_l'(A^l))
```

### Default simplification

In Phase 0–1, it is acceptable to use:

```text
eta_b = eta_w
```

but the code should not assume this permanently.

---

## 9. Recommended activation choices

Baseline defaults:

- hidden layers: `tanh` or `relu`
- output layer: `identity`

The activation derivative must be implemented explicitly.

### Important note

If `relu` is used, inference dynamics can become more brittle near zero. This is not forbidden, but experiments should document the choice.

---

## 10. Training algorithm (baseline)

For each batch `(x, y)`:

1. Clamp `X^0 = x`
2. Clamp `X^L = y`
3. Initialize free hidden states
4. Run `T_train` inference steps over free hidden layers
5. Recompute `A^l`, `MU^l`, `E^l`
6. Update all `W^l`, `b^l`
7. Log:
   - final energy
   - optional per-step energy
   - parameter norms

Pseudo-code:

```text
for each batch (x, y):
    X^0 = x
    X^L = y
    initialize X^1 ... X^(L-1)

    for t in 1..T_train:
        compute A, MU, E
        compute delta_X for free layers
        apply delta_X
        optionally record F_t

    compute A, MU, E one final time
    update W, b
```

---

## 11. Prediction algorithm (baseline)

For each input batch `x`:

1. Clamp `X^0 = x`
2. Initialize `X^1 ... X^L`
3. Run `T_eval` inference steps over free layers
4. Return `X^L`

Pseudo-code:

```text
X^0 = x
initialize X^1 ... X^L
for t in 1..T_eval:
    compute A, MU, E
    compute delta_X for free layers
    apply delta_X
return X^L
```

---

## 12. Numerical safeguards

The baseline implementation may include the following safeguards, but each must be documented in code:

- small weight initialization
- optional state clipping
- optional delta clipping
- optional damping factor on state updates
- NaN / Inf checks in tests or debug mode

Do not add undocumented stabilization tricks.

---

## 13. Phase 0 simplifications

The baseline implementation is allowed to make these simplifying choices:

- all `sigma_l^2 = 1`
- fixed inference step count `T`
- fixed learning rates
- no momentum / Adam
- no trainable recognition network
- no explicit probabilistic output likelihood beyond squared error

---

## 14. What must remain invariant unless the spec changes

The following are repository-level invariants for the baseline:

1. Arrays are batch-first
2. The baseline energy is squared-error over local prediction errors
3. Hidden states are inferred iteratively
4. Parameter updates are local functions of neighboring activities/errors under this spec
5. Clamped states are never updated during inference

---

## 15. Planned extension points

Likely future extensions include:

- separate recognition / initialization network
- alternative output likelihoods
- convolutional layers
- temporal predictive coding
- deeper-network stabilization variants
- alternative inference integrators

Those are not part of the baseline unless they are explicitly added in later versions of the spec.

---

## 16. Teacher-free FMPC v1 addendum

This addendum defines the first teacher-free FMPC transport contract without changing
the baseline predictive-coding energy, hidden-state gradient, or local parameter-update
mathematics.

### 16.1 Scope

Teacher-free FMPC v1 applies only to the existing layered predictive-coding substrate.

- it does not define a general PCG substrate
- it does not redefine predict-mode inference
- it does not replace the baseline iterative PC algorithm outside the explicit
  teacher-free transport experiment path

### 16.2 Training context and hidden latent

Teacher-free FMPC v1 uses the supervised training context:

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

Teacher-free FMPC v1 reuses the baseline predictive-coding energy:

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

Teacher-free FMPC v1 introduces an average-velocity model:

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

Teacher-free FMPC v1 uses the fixed-terminal-time direction:

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

Current TF2 contract note:

- the canonical TF2 default may keep the truncated feature-frozen identity
  approximation when matched validation runs do not show a stable empirical gain from
  feature-aware tangents

### 16.6 Parameter updates after transport

Teacher-free FMPC v1 changes only the hidden-state transport path used during training.

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


## 17. TF2 iFMPC bridge-stage addendum

This addendum defines an experimental **training-time scheduling extension** on top of
Phase TF1 without changing:

- the baseline predictive-coding energy
- the baseline hidden-state gradient definition
- the baseline local parameter-update equations
- the slow iterative predict/eval path

### 17.1 Scope

TF2 remains teacher-free and layered-PC-specific.

- it does not introduce a new substrate class
- it does not depend on JPC runtime
- it does not redefine predict-mode inference
- it does not add a new scaling mechanism in this first bridge pass

### 17.2 Micro-step schedule

Let `H = micro_steps` and define uniform rollout knots:

- `t_k = k / H`
- `-t = 1 / H`
- `r_k = 1 - t_k`

TF2 maintains two training-time hidden-state streams:

- `z_on_k`: learned on-policy hidden state
- `z_lf_k`: detached local-field-only shadow state

The state advances remain:

- `z_on_{k+1} = z_on_k + -t * u_psi(z_on_k, r_k, t_k; c)`
- `z_lf_{k+1} = z_lf_k + -t * g_theta(z_lf_k; c)`

where:

- `u_psi` is the same teacher-free average-velocity model from the TF1 addendum
- `g_theta(z; c) = --_z E_theta(z; c)` is unchanged

### 17.3 Frozen-within-micro-step semantics

Within a single micro-step `k`, all supervision targets and state advances must be
computed under one frozen parameter snapshot `(theta_k, psi_k)`.

This includes:

- `u_boot`
- `u_id`
- learned transport outputs
- `z_on_{k+1}`
- `z_lf_{k+1}`

Only after these quantities have been computed may parameter updates be applied.

The required order is:

1. compute supervision targets and learned transport under frozen `(theta_k, psi_k)`
2. advance `z_on` and `z_lf`
3. apply one immediate local `theta` update when enabled
4. apply one `psi` update

### 17.4 Mixed-policy teacher-free supervision

TF2 uses one of:

- `supervision_policy = "local_only"`
- `supervision_policy = "mixed"`

For `local_only`, `psi` is supervised only on detached `z_lf_k`.

For `mixed`, `psi` is supervised on the concatenation of detached:

- `z_lf_k`
- `z_on_k`

Targets remain the same TF1 teacher-free targets:

- `u_boot` from local self-bootstrap
- `u_id = g_t + r_k * D_T u_psi(...)`
- `L = L_boot + lambda_id * L_id`

If TF2 uses appended teacher-free current-state features in the psi input, then the
same two identity-tangent semantics from Section 16.5.1 apply:

- `feature_aware_tangents = true`
  - approximate the full augmented-input total derivative by injecting the feature
    directional-derivative block
- `feature_aware_tangents = false`
  - use the explicit truncated identity approximation that freezes the appended
    feature block inside the JVP path

### 17.5 Matched theta-update budget

TF2 introduces an explicit scheduling control:

- `theta_update_budget in {"matched", "unmatched"}`

If `incremental_weight_updates = true` and the budget is `matched`, the per-micro-step
parameter learning rates are normalized by the number of theta updates that are
actually applied under the active cadence for that batch:

- `theta_micro_lr = base_theta_lr / N_theta`
- `theta_micro_bias_lr = base_theta_bias_lr / N_theta`

where `N_theta` is:

- `1` for `terminal_only`
- `micro_steps` for `every_micro_step`
- the number of due theta-update events inside the micro-step loop for
  `every_2_micro_steps`

If `incremental_weight_updates = true` and the budget is `unmatched`, then:

- `theta_micro_lr = base_theta_lr`
- `theta_micro_bias_lr = base_theta_bias_lr`

If `incremental_weight_updates = false`, no theta updates occur inside the micro-step
loop and one terminal theta update is applied after the final micro-step using the
existing base learning rates.

This addendum changes only the **training-time schedule**. It does not redefine the
baseline local parameter-update rule itself.

### 17.6 Terminal local-field direction intervention

TF2 may optionally apply a **terminal-step teacher-free direction intervention**
during training:

- `terminal_local_field_direction_intervention in {`
  - `"none"`
  - `"local_field_direction_angle_clip_keep_live_norm"`
  - `"local_field_direction_hard_replace_keep_live_norm"`
  - `}`

This intervention is defined only for the **final micro-step** of the true closed-loop
training rollout. It does **not** change:

- the remaining-horizon contract for `u_psi(z_t, r, t; c)`
- the bootstrap target formula
- the identity target formula
- the evaluation-time transport operator

Let the raw learned terminal action be:

- `u_live = (z_{k+1}^{live} - z_k) / Δt`

and let the teacher-free local-field anchor direction be extracted from the current
terminal psi input:

- `d_lf = normalize(g_t)`

where `g_t` is the existing detached teacher-free local-flow block already present in
the TF2 input features.

If `terminal_local_field_direction_intervention = "none"`, TF2 uses:

- `u_term = u_live`

If `terminal_local_field_direction_intervention = "local_field_direction_hard_replace_keep_live_norm"`,
TF2 keeps the learned terminal norm but replaces the direction:

- `u_term = ||u_live|| * d_lf`

If `terminal_local_field_direction_intervention = "local_field_direction_angle_clip_keep_live_norm"`,
TF2 keeps the learned terminal norm but clips the learned terminal direction into a
cone around `d_lf` with half-angle `terminal_local_field_angle_clip_degrees`:

- `u_term = ||u_live|| * clip_dir(normalize(u_live), d_lf; theta_clip)`

The transported terminal state used for the immediate terminal theta update is then:

- `z_{k+1} = z_k + Δt * u_term`

This is a **training-time stabilization option** only. It preserves the TF2 teacher-free
target construction and keeps the historical corrective preset available as an
unstabilized reference.
