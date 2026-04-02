# Changelog

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

## phase1_5-stable

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
