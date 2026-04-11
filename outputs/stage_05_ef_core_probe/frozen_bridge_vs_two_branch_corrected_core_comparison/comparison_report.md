# Frozen Bridge vs Two-Branch Corrected Residual Core

## Protocol
- dataset: `digits`
- seeds: `[0, 1, 2]`
- shared batch size: `128`
- shared shuffle_batches: `True`

## Decision
- `stage05_v2_justifies_continued_exploration`: `True`
- `stage05_v2_as_new_exploratory_reference`: `True`
- `stage05_v2_replaces_frozen_bridge_on_main`: `False`
- rationale: `Stage 05 v2 clears the refreshed mechanism-first exploration rule, is stronger on one-step mechanism, is weaker on configured-step mechanism, and is weaker on report-only accuracy versus the frozen bridge.`

## Supports
- Stage 05 v2 artifacts are reproducible under the shared dataset/seed/batch protocol.
- Stage 05 v2 keeps one-step validation energy delta vs identity negative on every comparison seed.
- Stage 05 v2 keeps configured-step validation energy delta vs identity negative on every comparison seed.
- Stage 05 v2 keeps configured-step validation fixed-point residual delta vs identity negative on every comparison seed.
- Relative to the frozen bridge, Stage 05 v2 is stronger on one-step mechanism.
- Relative to the frozen bridge, Stage 05 v2 is weaker on configured-step mechanism.
- Relative to the frozen bridge, Stage 05 v2 is weaker on report-only accuracy.
- The refreshed comparison supports continued Stage 05 mechanism-first exploration.
- The refreshed comparison supports using Stage 05 v2 as the new exploratory reference.
- Stage 05 v2 mean validation accuracy is 0.270370; accuracy remains report-only in this comparison.

## Does Not Support
- This refreshed comparison supports keeping Stage 04 frozen on main.
- This refreshed comparison supports keeping Stage 05 mechanism-first.
- This refreshed comparison does not reopen any Stage 04 package-internal work.
- This refreshed comparison does not promote task accuracy to the Stage 05 gate.
- This refreshed comparison does not support replacing the frozen Stage 04 bridge on main.
