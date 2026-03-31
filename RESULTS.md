# Phase 2 Summary

This note freezes the repository's current Phase 2 state after the Phase 2e budget study.

It is intended as an internal research-engineering summary, not a paper-style claim set.

## Project State

Stable baseline and infrastructure:

- Phase 0: the baseline predictive-coding math is fixed by [spec_math.md](/e:/CodeSpace/PredictiveCoding/spec_math.md) and remains unchanged through Phase 2e.
- Phase 1: structured experiment outputs, saved traces, and benchmark scripts are in place.
- Phase 1.5: output layout semantics and seed semantics are explicit and stable.

Experimental findings layers:

- Phase 2a: minimal MLP baseline and direct PC-vs-MLP comparison.
- Phase 2b: narrow PC sensitivity analysis.
- Phase 2c: small multi-seed aggregate comparison.
- Phase 2d: diagnostics for the tuned-PC vs MLP gap.
- Phase 2e: tuned-PC inference-budget vs performance study.

## Phase 2a: Initial PC-vs-MLP Comparison

What was added:

- A minimal NumPy MLP baseline trained with ordinary backpropagation and SGD.
- A comparison runner that saves PC artifacts, MLP artifacts, and a comparison summary.

Question:

- How does the current predictive-coding baseline compare against a standard MLP on the current toy tasks?

Main findings:

- `toy_regression`: PC MSE `0.180622`, MLP MSE `0.000875`.
- `toy_sine_regression`: PC MSE `0.210139`, MLP MSE `0.125102`.
- `toy_blobs_classification`: PC accuracy `1.0`, MLP accuracy `1.0`, tie within tolerance.

Interpretation:

- On the two regression toy benchmarks, the initial PC baseline underperformed the MLP baseline.
- The regression gap was large enough to justify focused PC-only diagnostics before expanding model families or benchmark scope.

## Phase 2b: Small PC Sensitivity Analysis

What was added:

- A narrow one-at-a-time sensitivity layer over the current PC baseline.
- Controlled trials over `eta_x`, `eta_w`, `train_steps`, and `state_init`.

Question:

- Is the regression gap mainly due to conservative PC hyperparameters rather than the model family alone?

Main findings:

- On both regression benchmarks, the best small tuning result was `eta_w_double`.
- `toy_regression`: default PC MSE `0.180622` improved to `0.140146`.
- `toy_sine_regression`: default PC MSE `0.210139` improved to `0.190695`.
- In both cases, the tuned PC still remained behind the MLP reference.
- `state_init="zeros"` was clearly worse than the forward-initialized default on both regression tasks.

Interpretation:

- Modest tuning helps materially.
- The initial PC-vs-MLP gap was not just a trivial consequence of obviously bad default hyperparameters.

## Phase 2c: Multi-Seed Stability Check

What was added:

- A small multi-seed aggregate study over:
  - default PC
  - tuned PC using the fixed `eta_w_double` preset
  - MLP

Question:

- Is the tuned-PC improvement over default PC stable across seeds, or is it a one-off single-seed result?

Main findings:

- `toy_regression` over 5 seeds:
  - default PC mean MSE `0.160043`
  - tuned PC mean MSE `0.102322`
  - MLP mean MSE `0.000563`
  - tuned PC beat default PC on `5/5` seeds
  - MLP beat tuned PC on `5/5` seeds
- `toy_sine_regression` over 5 seeds:
  - default PC mean MSE `0.237242`
  - tuned PC mean MSE `0.201555`
  - MLP mean MSE `0.148488`
  - tuned PC beat default PC on `5/5` seeds
  - MLP beat tuned PC on `5/5` seeds

Interpretation:

- The tuned-PC improvement is stable.
- The tuned-PC deficit relative to MLP is also stable on the current regression toy tasks.

## Phase 2d: Diagnostics

What was added:

- A narrow diagnostic layer with:
  - per-epoch learning-curve aggregation
  - a `tuned_pc_budget2x` diagnostic branch
  - energy-vs-metric correlation summaries

Question:

- Why does tuned PC still trail MLP?

Main findings:

- `tuned_pc_budget2x` beat `tuned_pc` on both regression benchmarks across all 5 seeds:
  - `toy_regression`: tuned PC mean MSE `0.102322` to budget2x mean MSE `0.075322`
  - `toy_sine_regression`: tuned PC mean MSE `0.201555` to budget2x mean MSE `0.194871`
- For all three main variants, `best_epoch_mean` was the final epoch in the current training window, and `final_minus_best_metric_mean` was `0.0`.
- PC energy and task MSE were very tightly correlated on aggregated epoch-level mean curves:
  - roughly `0.99977` to `0.99998` on `toy_regression`
  - roughly `0.99967` to `0.99986` on `toy_sine_regression`

Interpretation:

- There is evidence that tuned PC still has an inference-budget bottleneck.
- There is not clear evidence that PC is minimizing a completely misaligned internal quantity:
  lower energy tracks lower task MSE closely on the current regression tasks.
- A more plausible current interpretation is that the gap to MLP is about optimization efficiency of the present PC training path, not about PC energy being unrelated to task performance.

## Phase 2e: Budget-Performance Tradeoff

What was added:

- A narrow budget study over:
  - `tuned_pc_1x`
  - `tuned_pc_2x`
  - `tuned_pc_4x`
  - fixed MLP reference

Question:

- If the tuned PC budget is increased further, how much does performance continue to improve, and does that close the gap to MLP?

Main findings:

- `toy_regression`:
  - tuned PC 1x mean MSE `0.102322`
  - tuned PC 2x mean MSE `0.075322`
  - tuned PC 4x mean MSE `0.051961`
  - MLP mean MSE `0.000563`
  - gap to MLP shrank from `0.101759` to `0.074759` to `0.051399`
  - `evidence_of_diminishing_returns = true`
- `toy_sine_regression`:
  - tuned PC 1x mean MSE `0.201555`
  - tuned PC 2x mean MSE `0.194871`
  - tuned PC 4x mean MSE `0.188202`
  - MLP mean MSE `0.148488`
  - gap to MLP shrank from `0.053067` to `0.046383` to `0.039714`
  - `evidence_of_diminishing_returns = true`
- Final PC energy also decreased monotonically with more budget on both regression tasks.

Interpretation:

- More inference budget keeps helping the tuned PC baseline.
- That improvement does not remove the gap to MLP on the current tasks.
- The budget curve shows diminishing returns, so simply increasing inference steps is not an unlimited path to closing the gap.

## Current Conclusions

- The baseline predictive-coding implementation is stable enough for narrow comparative studies.
- On the two current regression toy benchmarks, tuned PC is stably better than default PC.
- Tuned PC still stably underperforms the standard MLP baseline.
- Increasing tuned-PC inference budget from `1x` to `2x` to `4x` continues to improve performance.
- The Phase 2e aggregate summaries show diminishing returns.
- PC energy tracks task MSE closely on aggregated curves.
- The current best interpretation is:
  the remaining gap to MLP looks more like an optimization-efficiency gap than evidence that PC is optimizing a completely misaligned internal quantity.

## Limitations

- The stronger Phase 2 conclusions are only about:
  - `toy_regression`
  - `toy_sine_regression`
- The multi-seed studies keep `data_seed` fixed and vary `run_seed` with `model_init_seed`, so these phases mainly measure initialization stability rather than dataset-sampling variability.
- Phase 2e is not a wall-clock- or FLOP-matched efficiency comparison.
- No broad hyperparameter search was performed.
- No new model families or larger benchmarks were introduced.
- The diagnostics are suggestive, not causal proof.

## Recommended Next Steps

- Freeze the current repository state as a lightweight milestone before further exploratory work.
- If continuing within Phase 2-style analysis, prefer small studies that stay on the current toy regression tasks and directly target optimization efficiency rather than expanding benchmark scope.
- Possible next questions:
  - whether there is a useful budget range beyond `4x` before improvements saturate further
  - whether a better inference-efficiency schedule can improve PC without changing the Phase 0 math
  - whether the current gap is better explained by update efficiency per training epoch or per inference step

