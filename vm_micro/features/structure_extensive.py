"""
StructureBorneFeatureExtractorExtensive
=======================================
Separate class with redesigned feature extraction for structure-borne acoustic signals
in micro-drilling depth prediction.

Key improvements over V1:
  1. Fixed-window analysis (50 ms, 50 % overlap)  features decoupled from
     segment duration (eliminates the durationdepth confound).
  2. Higher effective sample rate (~48.8 kHz via cascade decimation 88)
      access to cutting-dynamics bands up to ~24 kHz.
  3. Normalised / intensive features (ratios, spectral shapes, slopes)
     instead of extensive quantities that scale with signal length.
  4. WPD sub-band energy ratios (uniform frequency resolution, replacing DWT).
  5. Cepstral features with log-spaced filterbank (MFCC-like, adapted for
     vibration rather than speech).
  6. Spectral-shape descriptors (slope, contrast, decrease, bandwidth ).
  7. CWT scale energy fractions (kept for comparability with airborne pipeline).
  8. Complexity measures (sample entropy, permutation entropy, Lempel-Ziv).

The returned dict contains *only* scalar feature values; meta-columns
(modality, record_name, depth_mm, ) must be added by the calling pipeline,
exactly as in the existing CSV schema.
"""

from __future__ import annotations

from collections import OrderedDict
from math import factorial, log2

import numpy as np
import pywt
from scipy.fft import rfft, rfftfreq
from scipy.signal import decimate
from scipy.signal import stft as _stft
from scipy.stats import kurtosis as sp_kurtosis
from scipy.stats import skew as sp_skew

#
# Main class
#


class StructureBorneFeatureExtractorExtensive:
    """
    Fixed-window, normalised feature extractor for structure-borne
    micro-drilling signals.

    Parameters
    ----------
    fs_native : int
        Native sampling rate of the raw signal (Hz).
    ds_stages : list[int]
        Cascade decimation factors.  Default [8, 8]  effective SR  48 828 Hz.
    window_s : float
        Analysis window length in seconds (default 0.050 = 50 ms).
    hop_s : float
        Hop size in seconds (default 0.025 = 25 ms, i.e. 50 % overlap).
    wpd_wavelet : str
        Wavelet for WPD (default 'db4').
    wpd_level : int
        WPD decomposition depth (default 5  32 sub-bands).
    n_mfcc : int
        Number of cepstral coefficients to retain (default 13).
    n_filters : int
        Number of triangular filters in the log-spaced filterbank (default 26).
    cwt_wavelet : str
        Wavelet for CWT (default 'morl').
    cwt_n_scales : int
        Number of CWT scales (default 32).
    cwt_f_min : float
        Lowest centre frequency for CWT scales (Hz, default 100).
    bands : list[tuple[float, float]]
        Frequency bands for band-energy-ratio features.
    complexity_n_samples : int
        Fixed number of samples used for complexity measures (default 5000).
    sampen_m / sampen_r_factor : int / float
        Sample-entropy embedding dimension and tolerance factor.
    permen_order / permen_delay : int / int
        Permutation-entropy parameters.
    """

    #  defaults
    _DEFAULTS = dict(
        ds_stages=[8, 8],
        window_s=0.050,
        hop_s=0.025,
        # WPD
        wpd_wavelet="db4",
        wpd_level=5,
        # Cepstral
        n_mfcc=13,
        n_filters=26,
        f_min_cepstral=50.0,
        # CWT
        cwt_wavelet="morl",
        cwt_n_scales=32,
        cwt_f_min=100.0,
        # Bands (Hz)
        bands=[
            (10, 500),
            (500, 2000),
            (2000, 5000),
            (5000, 10000),
            (10000, 25000),
        ],
        # Complexity
        complexity_n_samples=5000,
        sampen_m=2,
        sampen_r_factor=0.2,
        permen_order=5,
        permen_delay=1,
    )

    # Aggregation statistics applied to every per-window feature
    _AGG_NAMES = ("mean", "std", "med", "iqr", "slope", "rng")

    #  construction
    def __init__(self, fs_native: int, **kwargs):
        self.fs_native = int(fs_native)
        self.cfg = {**self._DEFAULTS, **kwargs}

        # Effective SR after cascade decimation
        self._ds_total = 1
        for f in self.cfg["ds_stages"]:
            self._ds_total *= f
        self.fs = self.fs_native // self._ds_total  # integer Hz

        # Window / hop in samples
        self.win_len = int(self.cfg["window_s"] * self.fs)
        self.hop_len = int(self.cfg["hop_s"] * self.fs)

        # Pre-build log-spaced triangular filterbank for cepstral features
        self._fbank = self._build_filterbank()

        # Pre-compute CWT scales (log-spaced from f_min to Nyquist)
        self._cwt_scales = self._build_cwt_scales()

    #  public API
    @property
    def ds_rate(self) -> int:
        return self._ds_total

    @property
    def sr_hz_used(self) -> int:
        return self.fs

    def extract(self, raw_data: np.ndarray) -> OrderedDict:
        signal = self._decimate(raw_data)
        windows = self._make_windows(signal)

        if len(windows) < 3:
            return self._empty_result()

        #  1. per-window feature vectors
        pw_keys: list[str] | None = None
        pw_matrix: list[list[float]] = []
        mfcc_matrix: list[np.ndarray] = []  # (n_windows, n_mfcc) for deltas

        for win in windows:
            row = OrderedDict()
            row.update(self._time_domain(win))
            row.update(self._spectral_shape(win))
            row.update(self._band_energy_ratios(win))
            row.update(self._wpd_ratios(win))
            mfcc_vec = self._cepstral(win)  # returns OrderedDict
            row.update(mfcc_vec)
            mfcc_matrix.append(np.array([mfcc_vec[k] for k in mfcc_vec]))
            if pw_keys is None:
                pw_keys = list(row.keys())
            pw_matrix.append([row[k] for k in pw_keys])

        pw_arr = np.array(pw_matrix, dtype=np.float64)  # (n_win, n_feat)
        mfcc_arr = np.array(mfcc_matrix, dtype=np.float64)  # (n_win, n_mfcc)

        #  2. aggregate per-window features
        result = self._aggregate(pw_keys, pw_arr)

        #  3. cepstral deltas (computed across the window axis)
        result.update(self._cepstral_deltas(mfcc_arr))

        #  4. CWT global features
        result.update(self._cwt_features(signal))

        #  5. STFT time-frequency dynamics
        result.update(self._timefrequency(signal))

        #  6. complexity on fixed-length excerpt
        result.update(self._complexity(signal))

        return result

    def feature_names(self, n_windows: int = 10) -> list[str]:
        dummy = np.random.randn(self.win_len * n_windows).astype(np.float32)
        return list(self.extract(dummy).keys())

    #  decimation
    def _decimate(self, raw: np.ndarray) -> np.ndarray:
        sig = raw.astype(np.float64).ravel()
        for factor in self.cfg["ds_stages"]:
            sig = decimate(sig, factor, ftype="iir", zero_phase=True)
        return sig

    #  windowing
    def _make_windows(self, signal: np.ndarray) -> list[np.ndarray]:
        wins = []
        start = 0
        while start + self.win_len <= len(signal):
            wins.append(signal[start : start + self.win_len])
            start += self.hop_len
        return wins

    #
    # PER-WINDOW FEATURE FAMILIES
    #

    #  A. time-domain
    @staticmethod
    def _time_domain(win: np.ndarray) -> OrderedDict:
        eps = 1e-15
        n = len(win)
        rms = np.sqrt(np.mean(win**2))
        peak = np.max(np.abs(win))
        crest = peak / (rms + eps)
        kurt = float(sp_kurtosis(win, fisher=True))
        skewn = float(sp_skew(win))
        zcr = np.sum(np.diff(np.sign(win)) != 0) / (n - 1)
        mean_abs = np.mean(np.abs(win))
        energy_ps = np.mean(win**2)  # energy per sample

        # Hjorth parameters
        diff1 = np.diff(win)
        diff2 = np.diff(diff1)
        activity = np.var(win)
        mobility_num = np.std(diff1)
        mobility = mobility_num / (np.std(win) + eps)
        complexity = (np.std(diff2) / (np.std(diff1) + eps)) / (mobility + eps)

        return OrderedDict(
            td_rms=rms,
            td_peak=peak,
            td_crest=crest,
            td_kurt=kurt,
            td_skew=skewn,
            td_zcr=zcr,
            td_mean_abs=mean_abs,
            td_energy_ps=energy_ps,
            td_hjorth_act=activity,
            td_hjorth_mob=mobility,
            td_hjorth_cmp=complexity,
        )

    #  B. spectral shape
    def _spectral_shape(self, win: np.ndarray) -> OrderedDict:
        eps = 1e-15
        N = len(win)
        # Hann-windowed FFT
        w = win * np.hanning(N)
        spec = np.abs(rfft(w))
        freqs = rfftfreq(N, d=1.0 / self.fs)

        # Power spectrum (one-sided)
        ps = spec**2
        ps_sum = ps.sum() + eps

        #  centroid
        centroid = np.sum(freqs * ps) / ps_sum
        #  spread
        spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * ps) / ps_sum)
        #  rolloff (95 %)
        cumsum = np.cumsum(ps)
        rolloff_idx = np.searchsorted(cumsum, 0.95 * cumsum[-1])
        rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]
        #  flatness (geometric mean / arithmetic mean of PS)
        log_ps = np.log(ps + eps)
        flatness = np.exp(log_ps.mean()) / (ps.mean() + eps)
        #  spectral slope (linear regression of log-PS vs freq)
        slope_coef = self._ols_slope(freqs, log_ps)
        #  spectral decrease
        if len(ps) > 1:
            decrease = np.sum((ps[1:] - ps[0]) / (np.arange(1, len(ps)) + eps)) / (
                ps_sum - ps[0] + eps
            )
        else:
            decrease = 0.0
        #  spectral skewness & kurtosis
        normed = (freqs - centroid) / (spread + eps)
        ss_skew = np.sum((normed**3) * ps) / ps_sum
        ss_kurt = np.sum((normed**4) * ps) / ps_sum
        #  spectral entropy
        p_norm = ps / ps_sum
        entropy = -np.sum(p_norm * np.log2(p_norm + eps))
        #  bandwidth at 3 dB & 10 dB
        peak_val = ps.max()
        bw3 = self._bandwidth_at_level(freqs, ps, peak_val, -3)
        bw10 = self._bandwidth_at_level(freqs, ps, peak_val, -10)
        #  peak frequency
        peak_freq = freqs[np.argmax(ps)]

        return OrderedDict(
            ss_centroid=centroid,
            ss_spread=spread,
            ss_rolloff=rolloff,
            ss_flatness=flatness,
            ss_slope=slope_coef,
            ss_decrease=decrease,
            ss_skew=ss_skew,
            ss_kurt=ss_kurt,
            ss_entropy=entropy,
            ss_bw3=bw3,
            ss_bw10=bw10,
            ss_peak_freq=peak_freq,
        )

    #  C. band energy ratios
    def _band_energy_ratios(self, win: np.ndarray) -> OrderedDict:
        eps = 1e-15
        N = len(win)
        w = win * np.hanning(N)
        spec = np.abs(rfft(w))
        ps = spec**2
        freqs = rfftfreq(N, d=1.0 / self.fs)
        total = ps.sum() + eps

        out = OrderedDict()
        for i, (lo, hi) in enumerate(self.cfg["bands"]):
            mask = (freqs >= lo) & (freqs < hi)
            ratio = ps[mask].sum() / total
            out[f"br_{i}"] = ratio
        return out

    #  D. WPD sub-band energy ratios
    def _wpd_ratios(self, win: np.ndarray) -> OrderedDict:
        eps = 1e-15
        level = self.cfg["wpd_level"]
        wavelet = self.cfg["wpd_wavelet"]
        try:
            wp = pywt.WaveletPacket(data=win, wavelet=wavelet, maxlevel=level)
        except Exception:
            n_nodes = 2**level
            return OrderedDict((f"wpd_{j:02d}", 0.0) for j in range(n_nodes))

        # Collect leaf-node energies in frequency order
        nodes = [node.path for node in wp.get_level(level, order="freq")]
        energies = np.array([np.sum(wp[p].data ** 2) for p in nodes], dtype=np.float64)
        total = energies.sum() + eps
        ratios = energies / total

        return OrderedDict((f"wpd_{j:02d}", ratios[j]) for j in range(len(ratios)))

    #  E. cepstral (MFCC-like, log-spaced filterbank)
    def _cepstral(self, win: np.ndarray) -> OrderedDict:
        n_mfcc = self.cfg["n_mfcc"]
        N = len(win)
        w = win * np.hanning(N)
        spec = np.abs(rfft(w))
        ps = spec**2

        # Apply filterbank   (n_filters,)
        fb_energies = self._fbank @ ps
        fb_energies = np.maximum(fb_energies, 1e-22)
        log_energies = np.log(fb_energies)

        # DCT-II to get cepstral coefficients
        n_filt = len(log_energies)
        dct_matrix = np.zeros((n_mfcc, n_filt))
        for k in range(n_mfcc):
            for m in range(n_filt):
                dct_matrix[k, m] = np.cos(np.pi * k * (2 * m + 1) / (2 * n_filt))
        coeffs = dct_matrix @ log_energies

        return OrderedDict((f"mfcc_{k:02d}", coeffs[k]) for k in range(n_mfcc))

    #
    # AGGREGATION across windows
    #

    def _aggregate(self, keys: list[str], arr: np.ndarray) -> OrderedDict:
        result = OrderedDict()
        for j, name in enumerate(keys):
            col = arr[:, j]
            m = np.nanmean(col)
            s = np.nanstd(col)
            med = np.nanmedian(col)
            q75, q25 = np.nanpercentile(col, [75, 25])
            iqr = q75 - q25
            sl = self._ols_slope_idx(col)
            rng = np.nanmax(col) - np.nanmin(col)
            for stat, val in zip(self._AGG_NAMES, [m, s, med, iqr, sl, rng]):
                result[f"{name}_{stat}"] = val
        return result

    #
    # CEPSTRAL DELTAS (across-window first & second derivatives)
    #

    def _cepstral_deltas(self, mfcc_arr: np.ndarray) -> OrderedDict:
        n_mfcc = mfcc_arr.shape[1]
        # delta: finite difference along window axis
        delta = np.diff(mfcc_arr, axis=0)
        ddelta = np.diff(delta, axis=0)

        result = OrderedDict()

        # delta features
        if delta.shape[0] >= 2:
            d_keys = [f"dmfcc_{k:02d}" for k in range(n_mfcc)]
            result.update(self._aggregate(d_keys, delta))
        else:
            for k in range(n_mfcc):
                for stat in self._AGG_NAMES:
                    result[f"dmfcc_{k:02d}_{stat}"] = np.nan

        # delta-delta features
        if ddelta.shape[0] >= 2:
            dd_keys = [f"ddmfcc_{k:02d}" for k in range(n_mfcc)]
            result.update(self._aggregate(dd_keys, ddelta))
        else:
            for k in range(n_mfcc):
                for stat in self._AGG_NAMES:
                    result[f"ddmfcc_{k:02d}_{stat}"] = np.nan

        return result

    #
    # CWT GLOBAL FEATURES  (on the full decimated segment)
    #

    def _cwt_features(self, signal: np.ndarray) -> OrderedDict:
        eps = 1e-15
        n_scales = self.cfg["cwt_n_scales"]
        try:
            coeffs, _ = pywt.cwt(
                signal,
                self._cwt_scales,
                self.cfg["cwt_wavelet"],
                sampling_period=1.0 / self.fs,
            )
        except Exception:
            out = OrderedDict()
            for i in range(n_scales):
                out[f"cwt_s{i:02d}_efrac"] = np.nan
            out["cwt_peak_scale"] = np.nan
            out["cwt_energy_spread"] = np.nan
            out["cwt_global_mean"] = np.nan
            out["cwt_global_std"] = np.nan
            return out

        abs_c = np.abs(coeffs)  # (n_scales, n_samples)
        scale_energy = np.sum(abs_c**2, axis=1)
        total_energy = scale_energy.sum() + eps
        efracs = scale_energy / total_energy

        out = OrderedDict()
        for i in range(n_scales):
            out[f"cwt_s{i:02d}_efrac"] = efracs[i]

        out["cwt_peak_scale"] = float(np.argmax(scale_energy))
        # energy spread: std of energy distribution across scales
        out["cwt_energy_spread"] = float(np.std(efracs))
        out["cwt_global_mean"] = float(np.mean(abs_c))
        out["cwt_global_std"] = float(np.std(abs_c))

        return out

    #
    # STFT-BASED TIME-FREQUENCY DYNAMICS  (on the full decimated segment)
    #

    def _timefrequency(self, signal: np.ndarray) -> OrderedDict:
        eps = 1e-15
        nperseg = min(2048, len(signal))
        hop = nperseg // 4

        freqs, times, Zxx = _stft(
            signal,
            fs=self.fs,
            nperseg=nperseg,
            noverlap=nperseg - hop,
        )
        mag = np.abs(Zxx)
        pwr = mag**2

        out = OrderedDict()

        if pwr.shape[1] < 2:
            out["tf_flux_mean"] = 0.0
            out["tf_flux_std"] = 0.0
            out["tf_flux_max"] = 0.0
            out["tf_variation_mean"] = 0.0
            out["tf_variation_median"] = 0.0
            out["tf_temporal_centroid"] = 0.5
            out["tf_dom_freq_mean"] = 0.0
            out["tf_dom_freq_std"] = 0.0
            out["tf_tonalness_mean"] = 0.0
            return out

        # Spectral flux
        flux = np.sqrt(np.sum(np.diff(mag, axis=1) ** 2, axis=0))
        out["tf_flux_mean"] = float(np.mean(flux))
        out["tf_flux_std"] = float(np.std(flux))
        out["tf_flux_max"] = float(np.max(flux))

        # Spectral variation (CV per frequency bin across time)
        sv = np.std(mag, axis=1) / (np.mean(mag, axis=1) + eps)
        out["tf_variation_mean"] = float(np.mean(sv))
        out["tf_variation_median"] = float(np.median(sv))

        # Temporal centroid of total power
        tp = np.sum(pwr, axis=0)
        t_norm = np.linspace(0, 1, len(tp))
        tp_norm = tp / (tp.sum() + eps)
        out["tf_temporal_centroid"] = float(np.sum(t_norm * tp_norm))

        # Dominant frequency tracking
        dom_f = freqs[np.argmax(pwr, axis=0)]
        out["tf_dom_freq_mean"] = float(np.mean(dom_f))
        out["tf_dom_freq_std"] = float(np.std(dom_f))

        # Tonalness (spectral flatness per frame, averaged)
        gm = np.exp(np.mean(np.log(pwr + eps), axis=0))
        am = np.mean(pwr + eps, axis=0)
        out["tf_tonalness_mean"] = float(np.mean(1.0 - gm / (am + eps)))

        return out

    #
    # COMPLEXITY MEASURES  (on a fixed-length excerpt)
    #

    def _complexity(self, signal: np.ndarray) -> OrderedDict:
        N = self.cfg["complexity_n_samples"]
        # Take a centred excerpt (or the whole signal if shorter)
        if len(signal) > N:
            start = (len(signal) - N) // 2
            excerpt = signal[start : start + N]
        else:
            excerpt = signal

        return OrderedDict(
            cx_sampen=self._sample_entropy(excerpt),
            cx_permen=self._permutation_entropy(excerpt),
            cx_lzc=self._lempel_ziv(excerpt),
        )

    #  sample entropy
    def _sample_entropy(self, x: np.ndarray) -> float:
        m = self.cfg["sampen_m"]
        r = self.cfg["sampen_r_factor"] * np.std(x)
        N = len(x)
        if N < m + 2 or r <= 0:
            return np.nan

        def _count(dim):
            templates = np.lib.stride_tricks.sliding_window_view(x, dim)
            n_t = len(templates)
            total = 0
            for i in range(n_t - 1):
                dists = np.max(np.abs(templates[i + 1 :] - templates[i]), axis=1)
                total += np.sum(dists < r)
            return total

        B = _count(m)
        A = _count(m + 1)
        if B == 0 or A == 0:
            return np.nan
        return -np.log(A / B)

    #  permutation entropy
    def _permutation_entropy(self, x: np.ndarray) -> float:
        order = self.cfg["permen_order"]
        delay = self.cfg["permen_delay"]
        n = len(x)
        max_perms = factorial(order)
        if n < (order - 1) * delay + 1:
            return np.nan

        counts: dict[tuple, int] = {}
        total = 0
        for i in range(n - (order - 1) * delay):
            motif = tuple(np.argsort(x[i : i + order * delay : delay]).tolist())
            counts[motif] = counts.get(motif, 0) + 1
            total += 1

        if total == 0:
            return np.nan
        probs = np.array(list(counts.values()), dtype=np.float64) / total
        H = -np.sum(probs * np.log2(probs + 1e-15))
        return H / log2(max_perms)  # normalised  [0, 1]

    #  Lempel-Ziv complexity (LZ76, normalised)
    @staticmethod
    def _lempel_ziv(x: np.ndarray) -> float:
        binary = (x > np.median(x)).astype(np.int8)
        n = len(binary)
        if n <= 1:
            return 0.0

        sub_strings: set[tuple] = set()
        w: list[int] = []
        c = 0
        for s in binary:
            w.append(int(s))
            if tuple(w) not in sub_strings:
                sub_strings.add(tuple(w))
                c += 1
                w = []
        if w:
            c += 1

        # Normalise: theoretical upper bound for binary sequence
        norm = n / log2(n + 1) if n > 1 else 1.0
        return c / norm

    #
    # HELPERS
    #

    def _build_filterbank(self) -> np.ndarray:
        n_filt = self.cfg["n_filters"]
        f_min = self.cfg["f_min_cepstral"]
        f_max = self.fs / 2.0
        n_fft = self.win_len

        # Log-spaced centre frequencies (n_filt + 2 points for edges)
        log_min = np.log10(max(f_min, 1.0))
        log_max = np.log10(f_max)
        centres = np.logspace(log_min, log_max, n_filt + 2)

        # Map to FFT bin indices
        n_bins = n_fft // 2 + 1
        bins = np.floor((n_fft + 1) * centres / self.fs).astype(int)
        bins = np.clip(bins, 0, n_bins - 1)

        fbank = np.zeros((n_filt, n_bins), dtype=np.float64)
        for i in range(n_filt):
            left, ctr, right = bins[i], bins[i + 1], bins[i + 2]
            if ctr == left:
                ctr = left + 1
            if right == ctr:
                right = ctr + 1
            # Rising slope
            for j in range(left, ctr):
                fbank[i, j] = (j - left) / (ctr - left)
            # Falling slope
            for j in range(ctr, right + 1):
                fbank[i, j] = (right - j) / (right - ctr)
        return fbank

    def _build_cwt_scales(self) -> np.ndarray:
        n = self.cfg["cwt_n_scales"]
        f_min = self.cfg["cwt_f_min"]
        f_max = self.fs / 2.0
        wavelet = self.cfg["cwt_wavelet"]

        # Centre frequency of the mother wavelet
        cf = pywt.central_frequency(wavelet)
        # scales = cf * fs / f    log-spaced f    log-spaced scales (reversed)
        freqs = np.logspace(np.log10(f_min), np.log10(f_max), n)
        scales = cf * self.fs / freqs
        return scales[::-1]  # ascending scale = descending frequency

    @staticmethod
    def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
        """OLS slope of y on x."""
        n = len(x)
        if n < 2:
            return 0.0
        xm = x.mean()
        denom = np.sum((x - xm) ** 2)
        if denom == 0:
            return 0.0
        return float(np.sum((x - xm) * (y - y.mean())) / denom)

    @staticmethod
    def _ols_slope_idx(col: np.ndarray) -> float:
        n = len(col)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        xm = x.mean()
        denom = np.sum((x - xm) ** 2)
        if denom == 0:
            return 0.0
        return float(np.sum((x - xm) * (col - col.mean())) / denom)

    @staticmethod
    def _bandwidth_at_level(freqs: np.ndarray, ps: np.ndarray, peak_val: float, db: float) -> float:
        threshold = peak_val * 10 ** (db / 10.0)
        mask = ps >= threshold
        if not np.any(mask):
            return 0.0
        idx = np.where(mask)[0]
        return float(freqs[idx[-1]] - freqs[idx[0]])

    def _empty_result(self) -> OrderedDict:
        # Build a dummy to get the keys
        dummy_win = np.random.randn(self.win_len)
        row = OrderedDict()
        row.update(self._time_domain(dummy_win))
        row.update(self._spectral_shape(dummy_win))
        row.update(self._band_energy_ratios(dummy_win))
        row.update(self._wpd_ratios(dummy_win))
        row.update(self._cepstral(dummy_win))
        pw_keys = list(row.keys())

        result = OrderedDict()
        for name in pw_keys:
            for stat in self._AGG_NAMES:
                result[f"{name}_{stat}"] = np.nan
        # delta / delta-delta
        for prefix in ("dmfcc", "ddmfcc"):
            for k in range(self.cfg["n_mfcc"]):
                for stat in self._AGG_NAMES:
                    result[f"{prefix}_{k:02d}_{stat}"] = np.nan
        # CWT
        for i in range(self.cfg["cwt_n_scales"]):
            result[f"cwt_s{i:02d}_efrac"] = np.nan
        result["cwt_peak_scale"] = np.nan
        result["cwt_energy_spread"] = np.nan
        result["cwt_global_mean"] = np.nan
        result["cwt_global_std"] = np.nan
        # Timefrequency
        for k in (
            "tf_flux_mean",
            "tf_flux_std",
            "tf_flux_max",
            "tf_variation_mean",
            "tf_variation_median",
            "tf_temporal_centroid",
            "tf_dom_freq_mean",
            "tf_dom_freq_std",
            "tf_tonalness_mean",
        ):
            result[k] = np.nan
        # Complexity
        result["cx_sampen"] = np.nan
        result["cx_permen"] = np.nan
        result["cx_lzc"] = np.nan
        return result


#
# Quick self-test
#
if __name__ == "__main__":
    import time

    FS_NATIVE = 3_125_000
    DURATION = 2.0  # seconds
    n_samples = int(FS_NATIVE * DURATION)

    print(f"Generating {DURATION}s synthetic signal at {FS_NATIVE / 1e6:.3f} MHz ")
    rng = np.random.default_rng(42)
    sig = rng.standard_normal(n_samples).astype(np.float32)
    # Add some tone structure
    t = np.arange(n_samples) / FS_NATIVE
    sig += 0.5 * np.sin(2 * np.pi * 3000 * t).astype(np.float32)
    sig += 0.3 * np.sin(2 * np.pi * 8000 * t).astype(np.float32)

    ext = StructureBorneFeatureExtractorExtensive(fs_native=FS_NATIVE)
    print(f"Effective SR: {ext.fs} Hz  (ds_rate = {ext.ds_rate})")
    print(f"Window: {ext.win_len} samples = {ext.cfg['window_s'] * 1000:.0f} ms")

    t0 = time.perf_counter()
    feats = ext.extract(sig)
    elapsed = time.perf_counter() - t0

    print(f"\nExtracted {len(feats)} features in {elapsed:.2f} s")
    print("\nFirst 20 features:")
    for i, (k, v) in enumerate(feats.items()):
        if i >= 20:
            break
        print(f"  {k:30s} = {v:+.6f}")

    # Check for NaNs
    n_nan = sum(1 for v in feats.values() if np.isnan(v))
    print(f"\nNaN count: {n_nan} / {len(feats)}")
