from __future__ import annotations

from functools import lru_cache
from math import ceil

import librosa
import numpy as np
import torch
import torch.nn as nn

try:
    import pywt
except Exception:
    pywt = None

from .config import TrainConfig


def _zscore_per_image(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return (x - mean) / std


def _power_to_db(power: torch.Tensor, amin: float = 1e-10) -> torch.Tensor:
    power = torch.clamp(power, min=amin)
    ref = power.amax(dim=(-2, -1), keepdim=True).clamp_min(amin)
    return 10.0 * (torch.log10(power) - torch.log10(ref))


@lru_cache(maxsize=16)
def _mel_filter_np(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    return librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=min(fmax, sample_rate / 2.0),
        htk=False,
        norm="slaney",
    ).astype(np.float32)


class TorchLogMelFrontend(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "mel_filter",
            torch.from_numpy(
                _mel_filter_np(
                    cfg.sample_rate,
                    cfg.n_fft,
                    cfg.n_mels,
                    cfg.fmin,
                    cfg.fmax,
                )
            ),
            persistent=True,
        )
        self.register_buffer(
            "window",
            torch.hann_window(cfg.n_fft, periodic=True),
            persistent=True,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            waveform.float(),
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.n_fft,
            window=self.window,
            center=False,
            return_complex=True,
        )
        power = spec.abs().pow(2.0)
        mel = torch.einsum("mf,bft->bmt", self.mel_filter, power)
        return _zscore_per_image(_power_to_db(mel)).unsqueeze(1)


def _morl_wavefun(precision: int) -> tuple[np.ndarray, np.ndarray]:
    length = 2**precision
    x = np.linspace(-8.0, 8.0, length, dtype=np.float64)
    psi = np.exp(-(x**2) / 2.0) * np.cos(5.0 * x)
    return psi, x


def _central_frequency_from_wavefun(psi: np.ndarray, x: np.ndarray) -> float:
    domain = float(x[-1] - x[0])
    idx = int(np.argmax(np.abs(np.fft.fft(psi))[1:]) + 2)
    if idx > len(psi) / 2:
        idx = len(psi) - idx + 2
    return 1.0 / (domain / (idx - 1))


def _integrate_wavelet(
    wavelet_name: str,
    precision: int = 10,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if pywt is not None:
        wavelet = pywt.DiscreteContinuousWavelet(wavelet_name)
        approx = wavelet.wavefun(precision)
        psi, x = approx if len(approx) == 2 else (approx[1], approx[-1])
        int_psi = np.cumsum(psi) * float(x[1] - x[0])
        if getattr(wavelet, "complex_cwt", False):
            int_psi = np.conj(int_psi)
        return np.asarray(int_psi), np.asarray(x), bool(getattr(wavelet, "complex_cwt", False))

    if wavelet_name != "morl":
        raise ImportError(
            "PyWavelets is not installed. Without it, only the default 'morl' wavelet is supported."
        )

    psi, x = _morl_wavefun(precision)
    return np.cumsum(psi) * float(x[1] - x[0]), x, False


def _wavelet_central_frequency(wavelet_name: str, precision: int = 8) -> float:
    if pywt is not None:
        return float(pywt.central_frequency(wavelet_name, precision=precision))

    if wavelet_name != "morl":
        raise ImportError(
            "PyWavelets is not installed. Without it, only the default 'morl' wavelet is supported."
        )

    psi, x = _morl_wavefun(precision)
    return float(_central_frequency_from_wavefun(psi, x))


def _next_fast_len(n: int) -> int:
    return int(2 ** np.ceil(np.log2(n)))


class TorchCWTFrontend(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg

        fmin = max(1.0, float(cfg.cwt_fmin))
        fmax = min(float(cfg.cwt_fmax), cfg.sample_rate / 2.0 - 1.0)
        freqs = np.geomspace(fmin, fmax, int(cfg.cwt_num_scales))
        scales = (
            _wavelet_central_frequency(cfg.cwt_wavelet, precision=cfg.cwt_precision)
            * cfg.sample_rate
            / freqs
        )

        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32), persistent=True)
        self._int_psi_np, self._x_np, self._complex_cwt = _integrate_wavelet(
            cfg.cwt_wavelet,
            cfg.cwt_precision,
        )

        self._bank_key: tuple[int, str] | None = None
        self.register_buffer("wavelet_fft_real", torch.zeros(1, 1), persistent=False)
        self.register_buffer("wavelet_fft_imag", torch.zeros(1, 1), persistent=False)
        self.register_buffer("crop_starts", torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer("scale_sqrt", torch.zeros(1), persistent=False)
        self._n_fft = 0

    def _build_bank(self, signal_len: int, device: torch.device) -> None:
        int_psi = torch.tensor(
            self._int_psi_np,
            device=device,
            dtype=torch.complex64 if np.iscomplexobj(self._int_psi_np) else torch.float32,
        )
        x = torch.tensor(self._x_np, device=device, dtype=torch.float32)
        step = x[1] - x[0]

        filters = []
        crop_starts = []
        conv_lengths = []

        for scale in self.scales.detach().cpu().numpy().tolist():
            support_len = int(ceil(scale * float(x[-1] - x[0]) + 1.0))
            indices = torch.floor(
                torch.arange(support_len, device=device, dtype=torch.float32)
                / (float(scale) * step)
            ).to(torch.long)
            indices = indices[indices < int_psi.numel()]
            filt = int_psi[indices].flip(0)
            filters.append(filt)
            conv_lengths.append(signal_len + int(filt.numel()) - 1)
            crop_starts.append(int(np.floor((filt.numel() - 2) / 2.0)))

        n_fft = _next_fast_len(max(conv_lengths))
        bank = torch.zeros((len(filters), n_fft), device=device, dtype=filters[0].dtype)
        for idx, filt in enumerate(filters):
            bank[idx, : filt.numel()] = filt

        bank_fft = torch.fft.fft(bank, dim=-1)
        self.wavelet_fft_real = bank_fft.real.to(torch.float32)
        self.wavelet_fft_imag = bank_fft.imag.to(torch.float32)
        self.crop_starts = torch.tensor(crop_starts, device=device, dtype=torch.long)
        self.scale_sqrt = torch.sqrt(self.scales.to(device=device, dtype=torch.float32))
        self._n_fft = n_fft
        self._bank_key = (signal_len, str(device))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.float()
        signal_len = waveform.shape[-1]

        if self._bank_key != (signal_len, str(waveform.device)):
            self._build_bank(signal_len, waveform.device)

        wavelet_fft = torch.complex(self.wavelet_fft_real, self.wavelet_fft_imag)
        fft_data = torch.fft.fft(waveform, n=self._n_fft, dim=-1)
        conv_full = torch.fft.ifft(fft_data[:, None, :] * wavelet_fft[None, :, :], dim=-1)
        coef_full = -self.scale_sqrt[None, :, None] * torch.diff(conv_full, dim=-1)

        coef = torch.stack(
            [
                coef_full[:, idx, start : start + signal_len]
                for idx, start in enumerate(self.crop_starts.tolist())
            ],
            dim=1,
        )

        if not self._complex_cwt:
            coef = coef.real

        power_db = 20.0 * torch.log10(coef.abs().clamp_min(1e-6))
        return _zscore_per_image(power_db).unsqueeze(1)


def build_frontend(cfg: TrainConfig) -> nn.Module:
    if cfg.feature_type == "logmel":
        return TorchLogMelFrontend(cfg)
    if cfg.feature_type == "cwt":
        return TorchCWTFrontend(cfg)
    raise ValueError(f"Unsupported feature_type: {cfg.feature_type}")
