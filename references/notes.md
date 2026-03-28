# Reference Notes

Use this file to record:

- which papers define the currently adopted formulation
- which equations from those papers map to `spec_math.md`
- implementation deviations from the papers
- open questions or ambiguities
- useful external repositories and what they are trusted for

Suggested structure:

## Primary formulation references

- Paper / section / equation numbers
- Why this paper is the authority for the current baseline

## Deviations in this repo

- Simpler initialization strategy
- MSE on one-hot labels in early classification
- Fixed-step inference in early phases

## Open questions

- Should a separate recognition network be introduced?
- Should output likelihood differ for classification?
- Which stabilization tricks are principled enough to admit into the baseline?
