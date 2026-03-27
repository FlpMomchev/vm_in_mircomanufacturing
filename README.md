# vm_in_micromanufacturing

Virtual metrology pipeline for micro-drilling depth prediction from acoustic sensing (airborne + structure-borne), with classical ML, deep learning, and multi-stage fusion.

## Current Status (2026-03-28)

- The full pipeline is implemented end-to-end:
  - split raw recordings into per-hole segments
  - extract modality-specific features
  - select features
  - train/evaluate classical models
  - train/evaluate DL models
  - run inference
  - fuse classical + DL per modality
  - fuse modalities into final prediction
- `vm-predict-fused` is available for one-shot production-style flow (`raw -> split -> infer -> fuse`).
- Automated tests exist for data I/O/splitting, features, classical, and fusion modules.

## Result Upload Policy

At this stage, do not upload new result dumps into arbitrary subfolders (for example under `models/**`).

Exception: `results/` is the intended place for curated, shareable benchmark exports when those are committed.

## Repository Structure

```text
vm_in_micromanufacturing/
  configs/
    paths.yaml
    split_presets.yaml
    airborne.yaml
    structure.yaml
    dl.yaml
    fusion.yaml
    predict_fused.yaml

  vm_micro/
    data/
      io.py
      manifest.py
      splitter.py
      plots.py
    features/
      core.py
      airborne.py
      structure.py
      structure_extensive.py
      selection.py
    classical/
      trainer.py
      inference.py
    dl/
      config.py
      data.py
      engine.py
      frontends.py
      models.py
      splits.py
      training.py
      utils.py
      visuals.py
    fusion/
      fuser.py
    utils/
      config.py
      paths.py
      logging.py

  scripts/
    split_audio.py      # vm-split
    extract_airborne.py # vm-extract-air
    extract_structure.py # vm-extract-struct
    select_features.py  # vm-select
    train_classical.py  # vm-train-cls
    train_dl.py         # vm-train-dl
    infer.py            # vm-infer
    fuse.py             # vm-fuse
    predict_fused.py    # vm-predict-fused

  docs/doe/
  data/
  models/
  tests/
  notebooks/
  .pre-commit-config.yaml
  .gitignore
  pyproject.toml
  README.md
```

## Current Best Metrics Snapshot

Metrics below are taken from the current artifact files in this repository.

### Classical (feature-based)

| Component | Artifact | MAE (mm) | RMSE (mm) | R2 |
|---|---|---:|---:|---:|
| Airborne top-3 ensemble | `models/features/air/final_models_fast_top3/final_model/ensemble_test_metrics.csv` | 0.0163 | 0.0443 | 0.9767 |
| Structure top-3 ensemble | `models/features/structure/final_models_fast/final_model/ensemble_test_metrics.csv` | 0.0260 | 0.0576 | 0.9606 |

### Deep Learning (repeat summary, mean test metrics)

| Component | Artifact | mean_test_mae (mm) | mean_test_rmse (mm) | mean_test_r2 |
|---|---|---:|---:|---:|
| Airborne SpecResNet (`BEST_MODEL`) | `models/dl/air/reg/air_spec_resnet_reg_BEST_MODEL/repeat_metrics_summary.json` | 0.0182 | 0.0799 | 0.9167 |
| Structure SpecResNet (`96k_retry`) | `models/dl/structure/reg/structure_spec_resnet_reg_96k_retry/repeat_metrics_summary.json` | 0.1395 | 0.1762 | 0.6310 |

### Fusion (labeled benchmark subset, n=18)

| Stage | Artifact | holdout_mae (mm) | holdout_rmse (mm) |
|---|---|---:|---:|
| Airborne intra-fusion | `models/fusion/airborne/fusion_report.json` | 0.0122 | 0.0219 |
| Structure intra-fusion | `models/fusion/structure/fusion_report.json` | 0.0104 | 0.0122 |
| Final inter-modality fusion | `models/fusion/final/fusion_report.json` | **0.0090** | 0.0125 |

Note: many files in `models/fusion/live_runs/**` are unlabeled production-style runs; holdout metrics there can be `0.0` because no ground truth is available.

## Installation

Requires Python 3.13+.

```bash
git clone <repo-url>
cd vm_in_micromanufacturing

python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev,dl]"
```

## CLI Commands

Installed console scripts:

- `vm-split`
- `vm-extract-air`
- `vm-extract-struct`
- `vm-select`
- `vm-train-cls`
- `vm-train-dl`
- `vm-infer`
- `vm-fuse`
- `vm-predict-fused`

## End-to-End Pipeline (Current)

### 1) Split raw recordings into per-hole segments

```bash
# Batch from named presets in configs/split_presets.yaml
vm-split batch --preset airborne_normalBand
vm-split batch --preset structure

# Single file example
vm-split single \
  --input data/raw_data/air_borne/normalBand/<recording>.flac \
  --out-dir data/raw_data_extracted_splits/air/<recording> \
  --segments-per-file 49 \
  --doe-xlsx docs/doe/Design_of_Experiment.xlsx
```

### 2) Extract features

```bash
# Airborne
vm-extract-air \
  --segments-dir data/raw_data_extracted_splits/air \
  --config configs/airborne.yaml \
  --out-csv data/features/airborne/features.csv

# Structure-borne (default extractor from configs/structure.yaml is "extensive")
vm-extract-struct \
  --segments-dir data/raw_data_extracted_splits/structure \
  --config configs/structure.yaml \
  --out-csv data/features/structure/features.csv
```

### 3) Feature selection

```bash
vm-select \
  --features-csv data/features/airborne/features.csv \
  --out-csv data/features/airborne/features_selected.csv \
  --final-n 20 \
  --min-partial-r 0.15
```

### 4) Train classical models

```bash
vm-train-cls \
  --features-csv data/features/airborne/features_selected.csv \
  --out-dir models/features/air/final_models_fast_top3 \
  --preset balanced \
  --ensemble-top-n 3

vm-train-cls \
  --features-csv data/features/structure/features_selected.csv \
  --out-dir models/features/structure/final_models_fast \
  --preset balanced \
  --ensemble-top-n 3
```

### 5) Train deep-learning models

```bash
# Airborne
vm-train-dl \
  --data-dir data/raw_data_extracted_splits/air \
  --output-dir models/dl/air/reg/air_spec_resnet_reg_BEST_MODEL \
  --task regression

# Structure
vm-train-dl \
  --data-dir data/raw_data_extracted_splits/structure \
  --output-dir models/dl/structure/reg/structure_spec_resnet_reg_96k_retry \
  --file-glob "**/*.h5" \
  --task regression \
  --model-type spec_resnet
```

### 6) Run modality inference

```bash
vm-infer classical \
  --bundle models/features/air/final_models_fast_top3/final_model/ensemble_model_bundle.joblib \
  --features data/features/airborne/features_selected.csv \
  --out-csv models/features/air/final_models_fast_top3/inference_predictions.csv

vm-infer dl \
  --model-dir models/dl/air/reg/air_spec_resnet_reg_BEST_MODEL \
  --data-dir data/raw_data_extracted_splits/air/live \
  --out-csv models/dl/air/reg/air_spec_resnet_reg_BEST_MODEL/inference_predictions.csv
```

### 7) Fuse predictions

```bash
# Intra-modality fusion (classical + DL)
vm-fuse intra \
  --classical-csv models/features/air/final_models_fast_top3/inference_predictions.csv \
  --classical-mae 0.0163 \
  --dl-csv models/dl/air/reg/air_spec_resnet_reg_BEST_MODEL/inference_predictions.csv \
  --dl-mae 0.0182 \
  --modality airborne_ensemble \
  --out-dir models/fusion/airborne

# Inter-modality fusion
vm-fuse inter \
  --bundle-csvs \
    models/fusion/airborne/fusion_predictions.csv:0.0122:airborne_ensemble \
    models/fusion/structure/fusion_predictions.csv:0.0104:structure_ensemble \
  --out-dir models/fusion/final
```

### 8) One-shot fused prediction from raw files

```bash
vm-predict-fused --config configs/predict_fused.yaml
```

This command scans configured raw folders, splits new files, runs modality inference, fuses outputs, and writes run artifacts to `models/fusion/live_runs/<timestamp>__<tag>/`.

## Fusion Architecture

```text
airborne FLAC segments
   vm-extract-air -> features.csv
        vm-select -> features_selected.csv
              vm-train-cls -> airborne_classical bundle
   vm-train-dl (logmel / CWT) -> airborne_dl bundle

       vm-fuse intra -> airborne_ensemble (w_cls, w_dl from inverse-MAE)

structure HDF5 segments
   vm-extract-struct (--extractor v1|extensive) -> features.csv
        vm-select (--min-partial-r 0.15) -> features_selected.csv
              vm-train-cls -> structure_classical bundle
   vm-train-dl -> structure_dl bundle

       vm-fuse intra -> structure_ensemble

                     vm-fuse inter -> final prediction + sigma_total
```

Weights are derived from inverse validation MAE (lower MAE gets higher weight).
Uncertainty is propagated as `sigma_total = sqrt(sum_i (w_i^2 * sigma_i^2))`.
The fusion module is self-contained in `vm_micro/fusion/fuser.py` and can be swapped (for example conformal intervals or a meta-learner) without changing extraction or training code.

## Extending The Pipeline

| Goal | Where to look |
|------|---------------|
| Add a new feature family | `vm_micro/features/core.py` (add function), then wire in `airborne.py` and/or `structure.py` |
| Change feature extraction parameters | `configs/airborne.yaml` or `configs/structure.yaml` |
| Add a new classical model | `vm_micro/classical/trainer.py` -> `make_model_specs()` |
| Add a new DL architecture | `vm_micro/dl/models.py` + register in `vm_micro/dl/frontends.py` |
| Change DL hyperparameters | `configs/dl.yaml` (all keys overridable via CLI) |
| Change fusion strategy | `vm_micro/fusion/fuser.py` -> `_fuse()` |
| Add or edit a batch split preset | `configs/split_presets.yaml` |
| Switch structure extractor | CLI `--extractor extensive` or `extractor: extensive` in `configs/structure.yaml` |
| Filter duration proxies from features | `vm-select --min-partial-r 0.15` |

## Testing

```bash
pytest -v
```

Tests cover:
- data IO and segmentation helpers
- feature extraction modules
- classical training/inference workflow
- fusion logic and report serialization

## Citation

*Filip Momchev, "Acoustic Sensor Fusion for Quality Prediction in Micro-Manufacturing: Evaluating Structure-Borne and Airborne Sound", Master's Thesis, Karlsruhe Institute of Technology, 2026.*
