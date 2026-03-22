# Air-influence exploration (legacy milling phase)

These four notebooks document the **milling phase** of the project, conducted
before switching to drilling holes as the target process.

| Notebook | Purpose |
|----------|---------|
| `01_first_visualization_analysis.ipynb` | First look at raw milling recordings — waveform, spectrogram, basic statistics. |
| `02_combined_analysis_with_cwt.ipynb` | CWT-based time-frequency analysis of milling signals. |
| `03_spectral_subtraction_improved.ipynb` | Spectral subtraction attempts to remove machine-air noise. |
| `04_ANC.ipynb` | Active Noise Cancellation experiments. |

## Why milling was abandoned

The machine-air supply introduced broadband noise that overwhelmed the
cutting signal in the 2–5 kHz band critical for depth estimation.  Spectral
subtraction and ANC could not reliably remove this artefact without distorting
depth-correlated features.

The decision was made to switch to **drilling holes** which allow the machine
air to be disabled, yielding clean airborne and structure-borne signatures.

These notebooks are preserved as methodological context and are **not** part
of the main processing pipeline.
