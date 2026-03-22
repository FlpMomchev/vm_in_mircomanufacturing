from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    """Configuration shared across training and inference."""

    data_dir: str
    file_glob: str = "**/*.flac"
    output_dir: str = "outputs_dl_depth_advanced"

    task: str = "classification"
    feature_type: str = "logmel"
    model_type: str = "hybrid_spec_transformer"

    sample_rate: int = 48_000
    window_sec: float = 0.50
    window_hop_sec: float = 0.25

    n_fft: int = 2_048
    hop_length: int = 512
    n_mels: int = 128
    fmin: float = 100.0
    fmax: float = 20_000.0

    cwt_wavelet: str = "morl"
    cwt_num_scales: int = 64
    cwt_fmin: float = 200.0
    cwt_fmax: float = 20_000.0
    cwt_precision: int = 10

    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5

    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True

    seed: int = 42
    min_windows_per_file: int = 1
    max_windows_per_file_train: int | None = None

    cache_audio: bool = True
    use_amp: bool = True
    device: str = "auto"
    channels_last: bool = True
    torch_compile: bool = False

    rounding_step_mm: float | None = 0.1

    # stronger model controls
    backbone_embed_dim: int = 128
    transformer_layers: int = 2
    transformer_token_grid_size: int = 8
    dropout: float = 0.20

    specaugment_time_masks: int = 2
    specaugment_time_width: int = 8
    specaugment_freq_masks: int = 2
    specaugment_freq_width: int = 12

    # LR scheduling and gradient clipping
    warmup_epochs: int = 2
    lr_min: float = 1e-5
    grad_clip: float = 1.0

    save_training_plots: bool = True
    save_attention_maps: bool = True
    attention_examples: int = 4

    def signal_num_samples(self) -> int:
        return int(round(self.window_sec * self.sample_rate))

    def ensure_output_dir(self) -> Path:
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), indent=2)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "TrainConfig":
        valid_fields = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in payload.items() if key in valid_fields}
        return cls(**filtered)
