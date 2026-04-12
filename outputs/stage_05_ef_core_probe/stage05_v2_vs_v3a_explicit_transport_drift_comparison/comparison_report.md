# Stage 05 v2 vs v3-A Explicit Transport-Drift Contract

## Protocol
- dataset: `digits`
- seeds: `[0]`
- shared batch size: `128`
- shared shuffle_batches: `True`

## Decision
- `stage05_v3a_shows_positive_gap_closure_signal_vs_v2`: `False`
- deterministic artifact checks all pass: `True`
- recommended next move: `run_fixed_budget_v2_vs_v3a_comparison`
- rationale: `The smoke-level v3-A candidate is artifact-stable and ready for a fixed-budget v2 vs v3-A comparison.`

## Supports
- The Stage 05 v3-A candidate path writes the standard Stage 05 artifacts.
- The v3-A candidate keeps artifact-independent target construction and the existing aggregate residual identity target.
- The smoke comparison exposes explicit pairwise deltas versus the current Stage 05 v2 reference.
- The v3-A smoke run passes deterministic artifact checks.

## Does Not Support
- This smoke comparison does not establish a formal fixed-budget mechanism win.
- This smoke comparison does not justify replacing the frozen Stage 04 bridge on main.
- The current evidence is still insufficient for a v2-to-v3-A adoption claim.
