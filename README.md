# vm_in_micromanufacturing

**Virtual Metrology for Micro-Drilling Depth Prediction via Acoustic Sensing**

Master's thesis project — predicts hole depth (0.1 mm – 1.0 mm in 0.1 mm steps)
from airborne and structure-borne acoustic signals recorded during micro-drilling,
using a combination of classical ML and deep-learning models fused into a single
continuous prediction with uncertainty.

---

## Repository structure

```
vm_in_micromanufacturing/
│
├── configs/                    # YAML configuration files
│   ├── paths.yaml              # Data / output path roots
│   ├── airborne.yaml           # Airborne feature extraction settings
│   ├── structure.yaml          # Structure-borne feature extraction settings
│   ├── dl.yaml                 # Deep-learning training hyper-parameters
│   └── fusion.yaml             # Fusion layer settings
│
├── vm_micro/                   # Main Python package
│   ├── data/
│   │   ├── io.py               # Unified FLAC / HDF5 reader
│   │   ├── manifest.py         # DOE loading, filename helpers
│   │   └── splitter.py         # Baseline-relative IDLE/ACTIVE segmenter
│   ├── features/
│   │   ├── core.py             # All feature families (time, freq, DWT, CWT …)
│   │   ├── airborne.py         # Airborne extraction pipeline (192 kHz FLAC)
│   │   ├── structure.py        # Structure-borne pipeline (HDF5, with decimation)
│   │   └── selection.py        # Inverted-cone feature selection
│   ├── classical/
│   │   ├── trainer.py          # Grouped CV training (RF, XGB, LGB, CatBoost, GPR …)
│   │   └── inference.py        # Bundle-based inference
│   ├── dl/                     # Deep-learning package (migrated from DL_depth_prediction)
│   │   ├── config.py           # TrainConfig dataclass
│   │   ├── data.py             # WaveformWindowDataset, AudioCache
│   │   ├── engine.py           # Train loop, predict, aggregate
│   │   ├── frontends.py        # Log-Mel and CWT frontends
│   │   ├── models.py           # SmallCNN, SpecResNet, HybridSpecTransformer
│   │   ├── splits.py           # Stratified grouped splits
│   │   ├── training.py         # fit_repeated_experiment, fit_final_model_all_files
│   │   ├── utils.py            # File table, label parsing, metrics
│   │   └── visuals.py          # Training plots, attention maps
│   ├── fusion/
│   │   └── fuser.py            # PredictionBundle, fuse_intra_modality, fuse_modalities
│   └── utils/
│       ├── config.py           # YAML loader + CLI override engine
│       ├── paths.py            # Project-root-relative path resolution
│       └── logging.py          # Logger factory
│
├── scripts/                    # CLI entry points (installed as `vm-*` commands)
│   ├── split_audio.py          # vm-split
│   ├── extract_airborne.py     # vm-extract-air
│   ├── extract_structure.py    # vm-extract-struct
│   ├── select_features.py      # vm-select
│   ├── train_classical.py      # vm-train-cls
│   ├── train_dl.py             # vm-train-dl
│   ├── infer.py                # vm-infer
│   └── fuse.py                 # vm-fuse
│
├── notebooks/
│   ├── air_influence_exploration/   # Legacy milling notebooks (01-04)
│   └── analysis/                    # side_effect_control, holdout_reliability …
│
├── tests/
│   ├── test_data_io.py         # I/O, manifest helpers, segmentation
│   ├── test_features.py        # Feature extraction correctness + edge cases
│   ├── test_classical.py       # Classical ML training + inference round-trip
│   └── test_fusion.py          # Fusion layer interface + uncertainty propagation
│
├── outputs/                    # Local model artefacts (git-ignored)
├── pyproject.toml
└── .gitignore
```

---

## Installation

Requires **Python 3.12+**.

```bash
git clone <your-repo-url>
cd vm_in_micromanufacturing

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install the package + all dependencies
pip install -e ".[dev]"
```

---

## Full pipeline walkthrough

### 1 · Segment recordings into per-hole files

```bash
# Single file
vm-split single \
    --input raw_data/airborne/0503_1_2_4532.flac \
    --out-dir all_outputs/0503_1_2_4532 \
    --segments-per-file 49

# Batch (all runs, with DOE mapping and canonical filenames)
vm-split batch --preset normalBand

# Custom band (for runs that need a different detection frequency)
vm-split batch --preset largerBand

# Structure-borne HDF5 (same CLI, auto-detected format)
vm-split single \
    --input raw_data/structure_borne/0503_1_2_4532.h5 \
    --out-dir all_outputs/structure/0503_1_2_4532 \
    --segments-per-file 49 \
    --band-low 200 --band-high 1000
```

### 2 · Extract features

```bash
# Airborne
vm-extract-air \
    --segments-dir all_outputs \
    --config       configs/airborne.yaml \
    --out-csv      outputs/features/airborne/features.csv

# Structure-borne
vm-extract-struct \
    --segments-dir all_outputs/structure \
    --config       configs/structure.yaml \
    --out-csv      outputs/structure/features.csv
```

### 3 · Select features

```bash
vm-select \
    --features-csv outputs/features/airborne/features.csv \
    --out-csv      outputs/features/airborne/features_selected.csv \
    --final-n      15
```

### 4 · Train classical models

```bash
vm-train-cls \
    --features-csv outputs/features/airborne/features_selected.csv \
    --out-dir      outputs/features/airborne \
    --holdout-runs 0303_3_1_8881 0503_7_2_9976
```

### 5 · Train DL models

```bash
# Classification
vm-train-dl \
    --data-dir    all_outputs \
    --output-dir  outputs/dl/hybrid_cls \
    --task        classification \
    --exclude-runs 0303_3_1_8881 0503_7_2_9976

# Regression
vm-train-dl \
    --data-dir    all_outputs \
    --output-dir  outputs/dl/hybrid_reg \
    --task        regression \
    --exclude-runs 0303_3_1_8881 0503_7_2_9976
```

### 6 · Run inference

```bash
vm-infer classical \
    --bundle   outputs/features/airborne/final_model/best_model_bundle.joblib \
    --features outputs/features/airborne/features_selected.csv \
    --out-csv  outputs/features/airborne/inference_predictions.csv

vm-infer dl \
    --model-dir outputs/dl/hybrid_cls \
    --data-dir  unseen \
    --out-csv   outputs/dl/hybrid_cls/inference_predictions.csv
```

### 7 · Fuse predictions

```bash
# Stage 1: airborne classical + DL
vm-fuse intra \
    --classical-csv outputs/features/airborne/inference_predictions.csv \
    --classical-mae 0.042 \
    --dl-csv        outputs/dl/hybrid_cls/inference_predictions.csv \
    --dl-mae        0.038 \
    --modality      airborne_ensemble \
    --out-dir       outputs/fusion/airborne

# Stage 2: airborne + structure-borne (once structure-borne is trained)
vm-fuse inter \
    --bundle-csvs \
        outputs/fusion/airborne/fusion_predictions.csv:0.040:airborne_ensemble \
        outputs/fusion/structure/fusion_predictions.csv:0.055:structure_ensemble \
    --out-dir outputs/fusion/final
```

---

## Running tests

```bash
pytest                  # all tests
pytest -v tests/test_fusion.py    # single module
pytest --co             # list collected tests without running
```

---

## Data layout (external, not tracked by Git)

```
raw_data/
  airborne/
    normalBand/    *.flac   (28 runs)
    largerBand/    *.flac   (2 runs)
  structure_borne/ *.h5     (future)
  Versuchsplan__Bohrungen.xlsx

all_outputs/       per-run subfolders, each containing:
  <run_stem>/
    <run_stem>__seg001__step001__B2__depth0.500.flac
    ...
    <run_stem>__debug__core.png
    <run_stem>__debug__padded.png

unseen/            held-out test files (0303_3_1_8881, 0503_7_2_9976)
inflated/          Audacity-augmented files for stress testing

outputs/           (git-ignored)
  dl/<model_tag>/final_model/best_model.pt
  features/<tag>/final_model/best_model_bundle.joblib
  fusion/<tag>/fusion_predictions.csv
```

---

## Fusion architecture

```
airborne FLAC segments
  ├── vm-extract-air  → features.csv
  │     └── vm-select → features_selected.csv
  │           └── vm-train-cls → airborne_classical bundle
  └── vm-train-dl (logmel / CWT) → airborne_DL bundle
            ↓                            ↓
       vm-fuse intra  →  airborne_ensemble  (w_cls, w_dl from inverse-MAE)
                                ↓
structure HDF5 segments  →  structure_ensemble  (future, same pipeline)
                                ↓
                     vm-fuse inter  →  final prediction + σ_total
```

Uncertainty is propagated as `σ_total = sqrt(Σ_i (w_i · σ_i)²)`.
The fusion module is fully self-contained in `vm_micro/fusion/fuser.py` and
can be swapped (conformal intervals, meta-learner, etc.) without touching
any training or extraction code.

---

## Extending the pipeline

| Goal | Where to look |
|------|---------------|
| Add a new feature family | `vm_micro/features/core.py` — add a function; wire it in `airborne.py` and/or `structure.py` |
| Add a new classical model | `vm_micro/classical/trainer.py` → `_build_models()` |
| Add a new DL architecture | `vm_micro/dl/models.py` + register in `frontends.py` |
| Change fusion strategy | `vm_micro/fusion/fuser.py` → `_fuse()` |
| Add a new batch preset | `scripts/split_audio.py` → `BATCH_PRESETS` |

---

## Citation / acknowledgements

*Filip [surname] — Master's thesis, [Institution], 2026.*
