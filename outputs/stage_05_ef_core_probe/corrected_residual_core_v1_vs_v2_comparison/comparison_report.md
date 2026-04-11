# Stage 05 Corrected Residual Core v1 vs v2

## Protocol
- dataset: `digits`
- seeds: `[0, 1, 2]`
- shared batch size: `128`
- shared shuffle_batches: `True`

## Decision
- `stage05_v2_improves_mechanism_magnitude_over_v1`: `True`
- rationale: `Stage 05 v2 improves mechanism magnitude over v1 under the narrow multiseed rule.`

## Supports
- Stage 05 v2 artifacts are reproducible under the shared dataset/seed/batch protocol.
- Stage 05 v2 keeps one-step validation energy delta vs identity negative on every comparison seed.
- Stage 05 v2 keeps configured-step validation fixed-point residual delta vs identity negative on every comparison seed.
- Stage 05 v2 improves mean configured-step validation energy delta vs identity over v1.
- Stage 05 v2 improves mean configured-step validation fixed-point residual delta vs identity over v1.
- The two-branch corrected residual core is favorable on mechanism-first grounds for the next narrow Stage 05 step.
- Stage 05 v1 mean validation accuracy is 0.274074 and Stage 05 v2 mean validation accuracy is 0.270370; accuracy remains report-only.

## Does Not Support
- This comparison does not claim that Stage 05 replaces the frozen Stage 04 bridge result on main.
- This comparison does not reopen any Stage 04 package-internal work.
- This comparison does not promote task accuracy to a gate.
