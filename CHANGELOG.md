# Changelog

## phase3-checkpoint

- Declared Phase 3 complete in the narrow sense of standalone real-data `digits` baselines
- Synced top-level docs to the current canonical Phase 3 artifact set:
  - `outputs/digits_mlp/`
  - `outputs/digits_pc/`
  - `outputs/digits_baselines/`
  - optional retained reference: `outputs/digits_pc_stabilization/`
- Removed lingering wording that implied the current Phase 3 state was still pending or undefined
- Removed the ambiguous `selected_candidate_rank` field from the standalone PC stabilization summary artifact
- Reframed the next phase as a controlled real-data comparison protocol, not a large immediate comparison push

## phase3-closure

- Synced Phase 3 docs with the current canonical `digits` artifacts:
  - `outputs/digits_mlp/`
  - `outputs/digits_pc/`
  - `outputs/digits_baselines/`
- Clarified that Phase 3 now means:
  - standalone `digits` MLP baseline
  - standalone `digits` PC baseline
  - protocol alignment checks
  - a first-pass side-by-side digest
- Clarified that Phase 3 still does not mean:
  - matched tuning
  - formal real-data comparison
  - multi-seed aggregation
  - a second real dataset
- Clarified that the next natural step is a cautious Phase 4 start, not a large immediate comparison push

## phase3b-pc-baseline

- Added a Phase 3 real-data predictive-coding baseline runner:
  - `src/pc/real_pc.py`
  - `experiments/digits_pc.py`
- Added protocol-alignment checks between `digits_mlp` and `digits_pc`
- Added a standalone side-by-side summary script:
  - `experiments/summarize_digits_baselines.py`
- Added a narrow stabilization sweep for the standalone `digits_pc` baseline
- Promoted the stabilization-selected best candidate to the canonical `digits_pc` default
- Added reproducible Phase 3 artifacts under:
  - `outputs/digits_pc/`
  - `outputs/digits_baselines/`
  - optional `outputs/digits_pc_stabilization/`

## phase3a-mlp-baseline

- Added a deterministic `digits` real-data split in `src/pc/datasets.py`
- Added a deterministic mini-batch helper in `src/pc/minibatch.py`
- Added a Phase 3 real-data MLP baseline runner:
  - `src/pc/real_mlp.py`
  - `experiments/digits_mlp.py`
- Kept the existing MLP math unchanged:
  - identity output
  - one-hot targets
  - MSE loss
- Added reproducible Phase 3 artifacts under `outputs/digits_mlp/`:
  - `config.json`
  - `epoch_metrics.csv`
  - `summary.json`
  - optional `plots/`
- Clarified in docs that Phase 3 currently means:
  - a small real-data MLP baseline exists
  - explicit `train / val / test` reporting exists
  - deterministic mini-batch ordering exists
- Clarified in docs that Phase 3 does not yet mean:
  - a real-data predictive-coding baseline
  - a real-data PC-vs-MLP comparison
  - an MNIST workflow
- Clarified usage/docs that the default `digits_mlp` run writes to the stable `outputs/digits_mlp/` directory and overwrites earlier files there on rerun

## historical phase1_5-stable baseline label

- The original `phase1_5-stable` git tag has been retired; this heading is now
  only a changelog label for the old baseline state.

- Stabilized Phase 0 predictive coding baseline
- Added structured experiment runner and output artifacts
- Added toy benchmark suite:
  - toy_regression
  - toy_sine_regression
  - toy_blobs_classification
- Clarified reproducibility semantics:
  - run_seed
  - data_seed
  - model_init_seed
- Added configurable output layout:
  - single_dir
  - run_id_subdir
