# Demo Deployment (Dashboard + Backend)

This document lists the minimum required components to run the dashboard and fused prediction pipeline on a new machine.

## 1) Required sources

### From the repository clone
The following tracked files are required:

- `app/**` (required)
- `scripts/final_prediction.py` (required)
- `vm_micro/**` (required)
- `configs/fusion.yaml` (required)
- `configs/airborne.yaml` (required)
- `configs/structure.yaml` (required)
- `dashboard_runtime/**` (required only if the app-only installation path is used)

### Transferred separately from the original machine
The following paths are ignored by Git and must be copied manually:

- `models/**` (required in practice; individual subfiles can be omitted only if the corresponding model family is disabled in config or fallback values are explicitly set)
- `data/features/**` (optional; can be omitted if target SR and extraction settings are pinned explicitly in config)
- `docs/doe/Experiment_DOE.xlsx` (optional; can be omitted if DOE mapping is disabled and expected segment defaults are set in config)

## 2) Required model and runtime artifacts

These files should be copied with the same relative paths.

### Airborne classical
- `models/features/air/final_model/final_model/ensemble_model_bundle.joblib` (required; cannot be reconstructed from config)
- `models/features/air/final_model/ensemble_test_predictions.csv` (optional; can be omitted if `models.airborne.classical.fusion_mae_fallback` is set in `configs/fusion.yaml`)

### Airborne DL
- `models/dl/air/reg/air_spec_resnet_reg_BEST_MODEL/final_model/best_model.pt` (required; cannot be reconstructed from config)
- `models/dl/air/reg/air_spec_resnet_reg_BEST_MODEL/final_model/config.json` (required; cannot be reconstructed from config)
- `models/dl/air/reg/air_spec_resnet_reg_BEST_MODEL/repeat_metrics_summary.json` (optional; can be omitted if `models.airborne.dl.fusion_mae_fallback` is set in `configs/fusion.yaml`)

### Structure classical
- `models/features/structure/default_features/final_models_balanced_v1/final_model/ensemble_model_bundle.joblib` (required; cannot be reconstructed from config)
- `models/features/structure/default_features/final_models_balanced_v1/ensemble_test_predictions.csv` (optional; can be omitted if `models.structure.classical.fusion_mae_fallback` is set in `configs/fusion.yaml`)

### Structure DL
- `models/dl/structure/reg/linear_spec_res_net_FINAL/final_model/best_model.pt` (required; cannot be reconstructed from config)
- `models/dl/structure/reg/linear_spec_res_net_FINAL/final_model/config.json` (required; cannot be reconstructed from config)
- `models/dl/structure/reg/linear_spec_res_net_FINAL/repeat_metrics_summary.json` (optional; can be omitted if `models.structure.dl.fusion_mae_fallback` is set in `configs/fusion.yaml`)

## 3) Recommended additional artifacts

These files are not the core model weights, but they improve runtime robustness and configuration recovery.

### Classical sidecars
- `models/features/air/final_model/final_model/best_model_metadata.json` (optional; can be omitted if extraction settings are pinned explicitly in `configs/airborne.yaml`)
- `models/features/air/final_model/run_config.json` (optional; can be omitted if extraction settings are pinned explicitly in `configs/airborne.yaml`)
- `models/features/structure/default_features/final_models_balanced_v1/final_model/best_model_metadata.json` (optional; can be omitted if extraction settings are pinned explicitly in `configs/structure.yaml`)
- `models/features/structure/default_features/final_models_balanced_v1/run_config.json` (optional; can be omitted if extraction settings are pinned explicitly in `configs/structure.yaml`)

### SR-sync fallback CSVs
- `data/features/structure/default_features/SELECTED_structure_features_v1.csv` (optional; can be omitted if target SR is pinned explicitly in `configs/structure.yaml`)
- `data/features/air/SELECTED_air_features.csv` (optional; can be omitted if target SR is pinned explicitly in `configs/airborne.yaml`)

### Batch-mode DOE mapping
- `docs/doe/Experiment_DOE.xlsx` (optional; can be omitted if `splitting.expected_segments.map_xlsx` is removed or set to `null` and a default segment count is configured)

## 4) Not required for runtime demo operation

The following are not needed for inference-only demo use:

- training outputs not used at inference time (not required)
- `models/fusion/**` reports (not required)
- notebooks (not required)
- development tooling (not required)

## 5) Installation

Python 3.13 is required.

From the repository root:

```powershell
pip install -e .\dashboard_runtime[dl]
```

## 6) Quick pre-flight check
```
$must = @(
  "configs/fusion.yaml",
  "configs/airborne.yaml",
  "configs/structure.yaml",
  "models/features/air/final_model/final_model/ensemble_model_bundle.joblib",
  "models/dl/air/reg/air_spec_resnet_reg_BEST_MODEL/final_model/best_model.pt",
  "models/dl/air/reg/air_spec_resnet_reg_BEST_MODEL/final_model/config.json",
  "models/features/structure/default_features/final_models_balanced_v1/final_model/ensemble_model_bundle.joblib",
  "models/dl/structure/reg/linear_spec_res_net_FINAL/final_model/best_model.pt",
  "models/dl/structure/reg/linear_spec_res_net_FINAL/final_model/config.json"
)
$must | ForEach-Object {
  if (Test-Path $_) { "OK`t$_" } else { "MISSING`t$_" }
}
```
## 7) Lauch commands
Fresh start:

```
vm-dashboard-run --fresh
```

Resume existing DB and history:
```
vm-dashboard-run
```

Fresh start and remove old displayed results:
```
vm-dashboard-run --fresh --purge-results
```
