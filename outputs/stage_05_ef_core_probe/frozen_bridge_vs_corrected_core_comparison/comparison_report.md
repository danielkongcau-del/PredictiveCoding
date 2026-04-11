# Frozen Bridge vs Corrected Residual Core

## Protocol
- dataset: `digits`
- seeds: `[0, 1, 2]`
- shared batch size: `128`
- shared shuffle_batches: `True`

## Decision
- `stage05_corrected_residual_core_justifies_v2_charter`: `True`
- rationale: `Stage 05 clears the multiseed mechanism-first comparison rule.`

## Supports
- Stage 05 comparison artifacts are reproducible under the shared dataset/seed/batch protocol.
- Stage 05 shows stable negative validation one-step energy delta vs identity across all comparison seeds.
- Stage 05 keeps configured-step validation fixed-point residual delta vs identity negative across all comparison seeds.
- The corrected residual core has enough mechanism-first signal to justify a Stage 05 v2 charter.
- Stage 05 report-only accuracy remains a contextual metric, not the gate, in this comparison.
- Stage 05 mean validation accuracy is 0.274074, which is reported but not used as the charter gate.

## Does Not Support
- This comparison does not promote Stage 05 to replace the frozen Stage 04 bridge on main.
- This comparison does not reopen any Stage 04 package-internal stabilizer search.
- This comparison does not claim that Stage 05 has solved the task-accuracy gap to the frozen bridge.
- Stage 05 remains below the frozen bridge on report-only test accuracy in the current comparison.
- Stage 05 does not outperform the frozen bridge on configured-step energy delta vs identity in this comparison.
