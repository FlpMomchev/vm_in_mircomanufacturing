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
│   │   ├── structure.py        # Structure-borne pipeline (HDF5, v1 + extensive routing)
│   │   ├── structure_extensive.py  # Windowed extractor (48.8 kHz, WPD, MFCC, complexity)
│   │   └── selection.py        # Inverted-cone feature selection (+ partial-r filter)
│   ├── classical/
│   │   ├── trainer.py          # Grouped CV training (RF, XGB, LGB, CatBoost, GPR …)
│   │   └── inference.py        # Bundle-based inference
│   ├── dl/                     # Deep-learning package
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
│   ├── extract_structure.py    # vm-extract-struct (--extractor v1|extensive)
│   ├── select_features.py      # vm-select (--min-partial-r for duration filtering)
│   ├── train_classical.py      # vm-train-cls
│   ├── train_dl.py             # vm-train-dl
│   ├── infer.py                # vm-infer
│   └── fuse.py                 # vm-fuse
│
├── results/                    # Benchmark results and plots (tracked by Git)
│   ├── airborne_classical/     # ExtraTrees holdout evaluation
│   ├── airborne_dl/            # SpecResNet unseen-run evaluation
│   └── RESULTS.md              # Full breakdown with all plots
│
├── notebooks/
│   ├── air_influence_exploration/       # Legacy milling notebooks (01-04)
│   ├── analysis/                        # side_effect_control, holdout_reliability …
│   ├── feature_extraction_validation.ipynb   # Per-modality feature QA
│   └── duration_dependency_diagnostic.ipynb  # Cross-modality duration confound analysis
│
├── tests/
│   ├── test_data_io.py         # I/O, manifest helpers, segmentation
│   ├── test_features.py        # Feature extraction correctness + new families
│   ├── test_classical.py       # Classical ML training + inference round-trip
│   └── test_fusion.py          # Fusion layer interface + uncertainty propagation
│
├── docs/doe/                   # DOE Excel manifests
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

## Design of Experiments (DOE)

**DOE1 — no air, depth sweep**
7×7 hole grid on a 97×97×11 mm plate, 12 mm pitch, 12.5 mm margins.
49 holes per plate run, drilled in randomised order.

| Parameter | Value |
|-----------|-------|
| Depth levels | 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 mm |
| Step size | 0.1 mm |
| Control depth | 0.5 mm (5 fixed control holes per plate) |
| Holes per run | 49 (44 test + 5 control) |
| Plate runs total | 31 (29 used for DL training, all 31 for classical) |

The run order is randomised per plate to decorrelate depth from spatial
position and tool wear progression. Control holes at fixed positions allow
drift correction across runs.

DOE files: [`docs/doe/`](docs/doe/)

---

## Results (airborne acoustics, current)

All evaluation (except for DL) uses **complete plate runs as holdout units** — the model never
sees any hole from a holdout run during training, feature selection, or
hyperparameter tuning. DL holdout evaluation is done **using single files as holdout units**,
because the model has no mechanism to memorise a run-level scalar.
This is the only protocol that reflects real deployment.

### Classical ML — ExtraTrees on 20 acoustic features

| Metric | Value |
|--------|-------|
| MAE | **0.032 mm** |
| RMSE | 0.046 mm |
| R² | 0.975 |
| P90 abs error | 0.071 mm |
| Nested CV MAE (31 runs, OOF) | 0.056 mm |
| Goal < 0.05 mm MAE | ✓ met |

Holdout: 7 complete plate runs, 343 holes, excluded from all phases.

> Structure-borne and fused results will be added as soon as those pipelines are
> completed.

---

## Running tests

```bash
pytest -v        # all tests, verbose
pytest --co      # list collected tests without running
```

**52 / 52 tests pass on a clean install** — 100 % pass rate.

The suite covers data I/O and segmentation (FLAC + HDF5 fixtures, full
detect → segment → pad → export cycle), all feature extraction families
with analytical ground-truth checks, classical ML training and inference
round-trip with a synthetic grouped dataset, and the full fusion layer
including weight computation, Gaussian uncertainty propagation, and record
intersection alignment.

---

## Full pipeline walkthrough

### 1 · Segment recordings into per-hole files

```bash
# Single file
vm-split single \
    --input raw_data/airborne/<recording>.flac \
    --out-dir all_outputs/<recording> \
    --segments-per-file 49

# Batch with DOE mapping and canonical filenames
vm-split batch --preset normalBand

# Custom detection band (adjust per recording if needed)
vm-split batch --preset largerBand

# Structure-borne HDF5 (format auto-detected)
vm-split single \
    --input raw_data/structure_borne/<recording>.h5 \
    --out-dir all_outputs/structure/<recording> \
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

# Structure-borne (default v1 extractor)
vm-extract-struct \
    --segments-dir all_outputs/structure \
    --config       configs/structure.yaml \
    --out-csv      outputs/structure/features.csv

# Structure-borne (extensive extractor — windowed, higher SR, WPD/MFCC)
vm-extract-struct \
    --segments-dir all_outputs/structure \
    --config       configs/structure.yaml \
    --out-csv      outputs/structure/features_extensive.csv \
    --extractor    extensive
```

### 3 · Select features

```bash
# Standard selection
vm-select \
    --features-csv outputs/features/airborne/features.csv \
    --out-csv      outputs/features/airborne/features_selected.csv \
    --final-n      15

# With duration-proxy filtering (genuine depth signal only)
vm-select \
    --features-csv <features_csv> \
    --out-csv      <output_csv> \
    --final-n      15 \
    --min-partial-r 0.15
```

### 4 · Train classical models

```bash
vm-train-cls \
    --features-csv <features_selected_csv> \
    --out-dir      <output_dir> \
    --holdout-runs <run_id_1> <run_id_2>
```

### 5 · Train DL models

```bash
# Classification
vm-train-dl \
    --data-dir    all_outputs \
    --output-dir  outputs/dl/hybrid_cls \
    --task        classification \
    --exclude-runs <run_id_1> <run_id_2>

# Regression
vm-train-dl \
    --data-dir    all_outputs \
    --output-dir  outputs/dl/hybrid_reg \
    --task        regression \
    --exclude-runs <run_id_1> <run_id_2>
```

### 6 · Run inference

```bash
vm-infer classical \
    --bundle   <model_bundle_path> \
    --features <features_selected_csv> \
    --out-csv  <output_predictions_csv>

vm-infer dl \
    --model-dir <dl_model_dir> \
    --data-dir  <data_dir> \
    --out-csv   <output_predictions_csv>
```

### 7 · Fuse predictions

```bash
# Stage 1: airborne classical + DL
vm-fuse intra \
    --classical-csv <classical_predictions_csv> \
    --classical-mae <value> \
    --dl-csv        <dl_predictions_csv> \
    --dl-mae        <value> \
    --modality      airborne_ensemble \
    --out-dir       outputs/fusion/airborne

# Stage 2: airborne + structure-borne (once structure-borne is trained)
vm-fuse inter \
    --bundle-csvs \
        <airborne_fusion_csv>:<mae>:airborne_ensemble \
        <structure_fusion_csv>:<mae>:structure_ensemble \
    --out-dir outputs/fusion/final
```

---

## Data layout (external, not tracked by Git)

```
raw_data/
  airborne/
    normalBand/    *.flac
    largerBand/    *.flac
  structure_borne/ *.h5
  Versuchsplan__Bohrungen.xlsx

all_outputs/       per-run subfolders, each containing segmented files:
  <run_stem>/
    <run_stem>__seg001__step001__<hole>__depth<d>.flac
    <run_stem>__debug__core.png
    <run_stem>__debug__padded.png

unseen/            held-out test recordings
inflated/          augmented files for stress testing

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
structure HDF5 segments
  ├── vm-extract-struct (--extractor v1|extensive)  → features.csv
  │     └── vm-select (--min-partial-r 0.15) → features_selected.csv
  │           └── vm-train-cls → structure_classical bundle
  └── vm-train-dl → structure_DL bundle
            ↓                            ↓
       vm-fuse intra  →  structure_ensemble
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
| Switch structure extractor | `configs/structure.yaml` → set `extractor: extensive` or use `--extractor extensive` |
| Filter duration proxies | `vm-select --min-partial-r 0.15` or set `min_partial_r` in `SelectionConfig` |

---

## Citation / acknowledgements

*Filip Momchev — Acoustic Sensor Fusion for Quality Prediction in Micro-Manufacturing: Evaluating Structure-Borne and Airborne Sound (Master's Thesis), Karlsruher Institut für Technologie, 2026.*
