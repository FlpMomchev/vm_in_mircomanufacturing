"""vm_micro.data.splitter
~~~~~~~~~~~~~~~~~~~~~~~~
Segment long recordings (FLAC or HDF5) into individual drilling events.

Algorithm
---------
1. Compute a 2–5 kHz band-power envelope (configurable via ``band_hz``).
2. Estimate a slow-drift baseline with an asymmetric EMA (resists being
   pulled up by long active plateaus).
3. Run an IDLE → ACTIVE → IDLE state machine on ``envelope – baseline``
   with hysteresis and minimum-persistence guards.
4. Post-process: split merged segments, merge chattering gaps, drop spikes.
5. Refine segment edges with a relaxed look-back / look-ahead pass.
6. Export segments with optional padding into FLAC or HDF5.

CLI entry point: ``vm-split``  (see scripts/split_audio.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

from .io import get_input_kind, read_signal_auto
from .manifest import (
    build_segment_filename,
    load_doe,
    load_expected_map_csv,
    map_segments_to_doe,
)


# ─────────────────────────────────────────────────────────────────────────────
# Envelope helpers
# ─────────────────────────────────────────────────────────────────────────────

def band_envelope_db(
    y: np.ndarray,
    sr: int,
    band_hz: tuple[float, float] = (2000.0, 5000.0),
    win_ms: float = 21.0,
    hop_ms: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, envelope_dB) for the specified frequency band."""
    nperseg  = max(256, int(sr * win_ms / 1000.0))
    noverlap = min(int(nperseg - max(1, int(sr * hop_ms / 1000.0))), nperseg - 1)

    f, t, Zxx = signal.stft(
        y, fs=sr, window="hann",
        nperseg=nperseg, noverlap=noverlap,
        detrend=False, boundary=None, padded=False,
    )
    P = (np.abs(Zxx) ** 2).astype(np.float64)
    lo, hi = band_hz
    band_mask = (f >= lo) & (f <= hi)
    band_power = P[band_mask].sum(axis=0) + 1e-18
    return t.astype(np.float64), (10.0 * np.log10(band_power)).astype(np.float64)


def _smooth_median(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x.copy()
    k = k if k % 2 else k + 1
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return signal.medfilt(xp, kernel_size=k)[pad:-pad]


def _rolling_quantile(x: np.ndarray, q: float, win: int) -> np.ndarray:
    if win <= 1:
        return x.copy()
    s = pd.Series(x)
    b = s.rolling(win, center=True, min_periods=max(3, win // 5)).quantile(q)
    b = b.interpolate(limit_direction="both").bfill().ffill()
    return b.to_numpy(dtype=np.float64)


def _asymmetric_baseline(b_raw: np.ndarray, win: int) -> np.ndarray:
    """Slow-rise / fast-fall EMA baseline; resists long active plateaus."""
    win = max(5, int(win))
    rise_alpha = 1.0 / (4.0 * win)
    fall_alpha = 1.0 / (1.0 * win)
    b = np.empty_like(b_raw, dtype=np.float64)
    b[0] = b_raw[0]
    for i in range(1, len(b_raw)):
        prev, cur = b[i - 1], b_raw[i]
        alpha = rise_alpha if cur > prev else fall_alpha
        b[i] = prev + alpha * (cur - prev)
    return b


# ─────────────────────────────────────────────────────────────────────────────
# State machine
# ─────────────────────────────────────────────────────────────────────────────

def _state_machine(
    delta_db: np.ndarray,
    times: np.ndarray,
    thr_on: float,
    thr_off: float,
    min_on_s: float,
    min_off_s: float,
) -> list[tuple[int, int]]:
    """IDLE/ACTIVE hysteresis state machine.  Returns frame-index pairs (i0, i1)."""
    if not len(delta_db):
        return []
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    on_frames  = max(1, int(round(min_on_s  / dt)))
    off_frames = max(1, int(round(min_off_s / dt)))

    IDLE, ACTIVE = 0, 1
    state = IDLE
    segs: list[tuple[int, int]] = []
    i0 = None
    on_count = off_count = 0

    for i, d in enumerate(delta_db):
        if state == IDLE:
            if d > thr_on:
                on_count += 1
                if on_count >= on_frames:
                    i0 = i - on_frames + 1
                    state = ACTIVE
                    off_count = on_count = 0
            else:
                on_count = 0
        else:
            if d < thr_off:
                off_count += 1
                if off_count >= off_frames:
                    end = i - off_frames
                    if i0 is not None and end >= i0:
                        segs.append((i0, end))
                    state = IDLE
                    i0 = None
                    off_count = 0
            else:
                off_count = 0

    if state == ACTIVE and i0 is not None:
        segs.append((i0, len(delta_db) - 1))
    return segs


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _seg_duration(times: np.ndarray, seg: tuple[int, int]) -> float:
    return float(times[seg[1]] - times[seg[0]])


def _seg_strength(delta_db: np.ndarray, seg: tuple[int, int]) -> float:
    a, b = seg
    return float(np.max(delta_db[a : b + 1])) if b >= a else -np.inf


def _split_at_valley(
    delta_db: np.ndarray,
    times: np.ndarray,
    seg: tuple[int, int],
    thr_off: float,
    min_valley_s: float,
) -> list[tuple[int, int]]:
    a, b = seg
    if b - a < 10:
        return [seg]
    d, t = delta_db[a : b + 1], times[a : b + 1]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.01
    min_sep = max(1, int(round(min_valley_s / dt)))
    peaks, _ = signal.find_peaks(d, distance=min_sep)
    if len(peaks) < 2:
        return [seg]
    order = np.argsort(d[peaks])[::-1]
    p1, p2 = sorted([peaks[order[0]], peaks[order[1]]])
    if p2 - p1 < min_sep:
        return [seg]
    valley_rel = int(np.argmin(d[p1 : p2 + 1]) + p1)
    if float(d[valley_rel]) >= thr_off:
        return [seg]
    cut = a + valley_rel
    left  = (a, max(a, cut))
    right = (min(b, cut + 1), b)
    if left[1] - left[0] < 3 or right[1] - right[0] < 3:
        return [seg]
    return [left, right]


def _split_all_valleys(
    segs: list[tuple[int, int]],
    delta_db: np.ndarray,
    times: np.ndarray,
    thr_off: float,
    min_valley_s: float,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s in segs:
        out.extend(_split_at_valley(delta_db, times, s, thr_off, min_valley_s))
    return sorted(out, key=lambda s: s[0])


def _merge_close(
    segs: list[tuple[int, int]],
    times: np.ndarray,
    merge_gap_s: float,
) -> list[tuple[int, int]]:
    if not segs:
        return []
    segs = sorted(segs, key=lambda s: s[0])
    out = [segs[0]]
    for a, b in segs[1:]:
        pa, pb = out[-1]
        if float(times[a] - times[pb]) <= merge_gap_s:
            out[-1] = (pa, max(pb, b))
        else:
            out.append((a, b))
    return out


def _drop_short(
    segs: list[tuple[int, int]],
    delta_db: np.ndarray,
    times: np.ndarray,
    min_dur_s: float,
) -> list[tuple[int, int]]:
    return [s for s in segs if _seg_duration(times, s) >= min_dur_s]


def _refine_to_expected(
    segs: list[tuple[int, int]],
    delta_db: np.ndarray,
    times: np.ndarray,
    expected_n: int,
) -> list[tuple[int, int]]:
    """Iteratively drop spurious or split merged segments to reach expected_n."""
    if expected_n is None:
        return segs
    segs = list(segs)

    if len(segs) > expected_n:
        durations  = np.array([_seg_duration(times, s)  for s in segs], dtype=float)
        strengths  = np.array([_seg_strength(delta_db, s) for s in segs], dtype=float)
        score = durations + 0.15 * (strengths - np.median(strengths))
        keep  = np.ones(len(segs), dtype=bool)
        for idx in np.argsort(score):
            if keep.sum() <= expected_n:
                break
            keep[idx] = False
        segs = [s for k, s in zip(keep, segs) if k]

    if len(segs) < expected_n and segs:
        typical     = float(np.median([_seg_duration(times, s) for s in segs]))
        typical     = typical if np.isfinite(typical) and typical > 0 else 0.5
        min_valley_s = max(0.10, 0.25 * typical)
        thr_off      = float(np.percentile(delta_db, 35))

        changed = True
        while changed and len(segs) < expected_n:
            changed = False
            segs = sorted(segs, key=lambda s: _seg_duration(times, s), reverse=True)
            new_segs: list[tuple[int, int]] = []
            for s in segs:
                parts = _split_at_valley(delta_db, times, s, thr_off, min_valley_s)
                if len(parts) == 2 and len(new_segs) + len(segs) - 1 < expected_n:
                    new_segs.extend(parts)
                    changed = True
                else:
                    new_segs.append(s)
            segs = sorted(new_segs, key=lambda s: s[0])

    return segs


# ─────────────────────────────────────────────────────────────────────────────
# Main detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_segments(
    y: np.ndarray,
    sr: int,
    segments_per_file: int = 10,
    band_hz: tuple[float, float] = (2000.0, 5000.0),
    win_ms: float = 21.0,
    hop_ms: float = 5.0,
    smooth_win_s: float | None = None,
    baseline_q: float = 0.10,
    baseline_win_s: float | None = None,
    min_on_frac: float  = 0.08,
    min_off_frac: float = 0.08,
    merge_gap_frac: float   = 0.05,
    min_valley_frac: float  = 0.20,
    min_keep_frac: float    = 0.10,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    """Detect drilling segments using baseline-relative IDLE/ACTIVE machine.

    Returns
    -------
    segs_s : list of (start_s, end_s) in seconds
    dbg    : debug dict for plotting and edge refinement
    """
    times, env_db = band_envelope_db(y, sr, band_hz=band_hz, win_ms=win_ms, hop_ms=hop_ms)

    dur = float(times[-1]) if len(times) else 0.0
    edge = min(0.25, 0.05 * dur) if dur > 0 else 0.0
    valid = (times >= edge) & (times <= (dur - edge))
    if valid.sum() < 10:
        valid = np.ones_like(times, dtype=bool)

    typical_s = dur / float(segments_per_file) if segments_per_file and dur > 0 else 0.5
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.01

    if smooth_win_s is None:
        smooth_win_s = max(0.03, 0.08 * typical_s)
    k = max(1, int(round(smooth_win_s / dt)))
    env_s = _smooth_median(env_db, k)

    if baseline_win_s is None:
        baseline_win_s = max(0.5, 0.8 * typical_s)
    base_win = max(5, int(round(baseline_win_s / dt)))
    baseline = _asymmetric_baseline(_rolling_quantile(env_s, baseline_q, base_win), base_win)

    delta   = env_s - baseline
    d_valid = delta[valid]

    iqr     = float(np.subtract(*np.percentile(d_valid, [75, 25])) + 1e-6)
    hyst_db = max(0.8, 0.20 * iqr)

    min_on_s     = max(0.03, min_on_frac     * typical_s)
    min_off_s    = max(0.03, min_off_frac    * typical_s)
    merge_gap_s  = max(0.02, merge_gap_frac  * typical_s)
    min_valley_s = max(0.08, min_valley_frac * typical_s)
    min_keep_s   = max(0.08, min_keep_frac   * typical_s)

    thr_lo = float(np.percentile(d_valid, 2))
    thr_hi = float(np.percentile(d_valid, 99.7))

    best: tuple[int, float, float, list] | None = None

    for _ in range(32):
        thr_on  = 0.5 * (thr_lo + thr_hi)
        thr_off = thr_on - hyst_db

        segs = _state_machine(delta, times, thr_on, thr_off, min_on_s, min_off_s)
        segs = _split_all_valleys(segs, delta, times, thr_off, min_valley_s)
        segs = _merge_close(segs, times, merge_gap_s)
        segs = _drop_short(segs, delta, times, min_keep_s)
        if segments_per_file is not None and len(segs) < segments_per_file:
            segs = _refine_to_expected(segs, delta, times, segments_per_file)

        n = len(segs)
        if best is None or abs(n - segments_per_file) < abs(best[0] - segments_per_file):
            best = (n, thr_on, thr_off, segs)
        if n > segments_per_file:
            thr_lo = thr_on
        elif n < segments_per_file:
            thr_hi = thr_on
        else:
            best = (n, thr_on, thr_off, segs)
            break

    assert best is not None
    _, thr_on, thr_off, segs = best
    segs_s = [(float(times[a]), float(times[b])) for a, b in segs]

    dbg: dict[str, Any] = dict(
        times=times, env_db=env_db, env_s=env_s, baseline=baseline, delta=delta,
        thr_on=thr_on, thr_off=thr_off, typical_s=typical_s,
        smooth_win_s=smooth_win_s, baseline_win_s=baseline_win_s,
        min_on_s=min_on_s, min_off_s=min_off_s,
        merge_gap_s=merge_gap_s, min_valley_s=min_valley_s, min_keep_s=min_keep_s,
    )
    return segs_s, dbg


def refine_segment_edges(
    dbg: dict[str, Any],
    segs_s: list[tuple[float, float]],
    lookback_s: float = 3.0,
    lookahead_s: float = 0.25,
    on_ratio: float = 0.50,
    off_ratio: float = 1.00,
    snap_min_window_s: float = 0.08,
) -> list[tuple[float, float]]:
    """Adjust segment edges using a relaxed threshold within limited windows."""
    t    = np.asarray(dbg["times"])
    env  = np.asarray(dbg["env_s"])
    base = np.asarray(dbg["baseline"])
    thr_on  = float(dbg["thr_on"])
    thr_off = float(dbg["thr_off"])
    dt   = float(np.median(np.diff(t)))
    n    = len(t)
    lb   = int(round(lookback_s / dt))
    la   = int(round(lookahead_s / dt))
    snap = int(round(snap_min_window_s / dt))

    thr_low_on  = base + max(0.2, thr_on  * on_ratio)
    thr_low_off = base + max(0.2, thr_off * off_ratio)

    def sec_to_idx(x: float) -> int:
        return int(np.clip(np.searchsorted(t, x, side="left"), 0, n - 1))

    refined: list[tuple[float, float]] = []
    prev_end = 0

    for a, b in segs_s:
        i0 = sec_to_idx(a)
        i1 = sec_to_idx(b)

        j = i0
        j_min = max(prev_end, i0 - lb)
        while j > j_min and env[j] > thr_low_on[j]:
            j -= 1
        w0 = max(prev_end, j - snap)
        w1 = min(i0, j + snap)
        if w1 > w0:
            j = w0 + int(np.argmin(env[w0:w1]))

        k = i1
        k_max = min(n - 1, i1 + la)
        while k < k_max and env[k] > thr_low_off[k]:
            k += 1

        j = min(j, k - 1)
        refined.append((float(t[j]), float(t[k])))
        prev_end = k

    return refined


def apply_padding(
    segs_s: list[tuple[float, float]],
    pre_pad_s: float,
    post_pad_s: float,
    duration_s: float,
) -> list[tuple[float, float]]:
    out = []
    for a, b in segs_s:
        a2 = max(0.0, a - float(pre_pad_s))
        b2 = min(float(duration_s), b + float(post_pad_s))
        if b2 > a2:
            out.append((a2, b2))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def _seg_to_samples(
    a: float, b: float, sr: int, n_samples: int
) -> tuple[int, int]:
    s0 = max(0, min(n_samples, int(round(a * sr))))
    s1 = max(0, min(n_samples, int(round(b * sr))))
    return s0, s1


def _make_time_vector(n: int, sr: int, offset_s: float = 0.0) -> np.ndarray:
    return offset_s + np.arange(n, dtype=np.float64) / float(sr)


def _normalize_for_audio(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - float(np.mean(x))
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return x / m if m > 0 else x


def _write_h5_segment(
    out_path: Path,
    data: np.ndarray,
    time_vector: np.ndarray,
    data_key: str,
    time_key: str,
    attrs: dict[str, Any] | None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    group, dname = data_key.rsplit("/", 1)
    _, tname = time_key.rsplit("/", 1)
    with h5py.File(out_path, "w") as fh:
        grp = fh.require_group(group)
        grp.create_dataset(dname, data=np.asarray(data))
        grp.create_dataset(tname, data=np.asarray(time_vector, dtype=np.float64))
        if attrs:
            for k, v in attrs.items():
                try:
                    grp.attrs[k] = v
                except Exception:
                    grp.attrs[k] = str(v)


def export_segments(
    y: np.ndarray,
    sr: int,
    segs_s: list[tuple[float, float]],
    out_dir: Path,
    filenames: list[str],
    *,
    export_format: str,
    input_kind: str,
    time_vector: np.ndarray | None = None,
    h5_data_key: str = "measurement/data",
    h5_time_key: str = "measurement/time_vector",
) -> list[Path]:
    """Write one file per segment; return output paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(filenames) != len(segs_s):
        raise ValueError("filenames and segs_s must have the same length")

    fmt = _resolve_fmt(export_format, input_kind)
    tv_full = np.asarray(time_vector, dtype=np.float64) if time_vector is not None else None
    out_paths: list[Path] = []

    for (a, b), name in zip(segs_s, filenames):
        s0, s1 = _seg_to_samples(a, b, sr, len(y))
        chunk = np.asarray(y[s0:s1])
        p = out_dir / name

        if fmt in {"flac", "wav"}:
            audio = chunk if input_kind == "audio" else _normalize_for_audio(chunk)
            sf.write(p, audio, sr)
        elif fmt == "h5":
            tv = tv_full[s0:s1] if tv_full is not None else _make_time_vector(len(chunk), sr, a)
            _write_h5_segment(
                p, chunk, tv, h5_data_key, h5_time_key,
                attrs={"sample_rate_hz": int(sr), "segment_start_s": a, "segment_end_s": b},
            )
        elif fmt == "npz":
            tv = tv_full[s0:s1] if tv_full is not None else _make_time_vector(len(chunk), sr, a)
            np.savez(p, data=chunk, time_vector=tv, sample_rate_hz=int(sr),
                     segment_start_s=a, segment_end_s=b)
        out_paths.append(p)

    return out_paths


def _resolve_fmt(export_format: str, input_kind: str) -> str:
    fmt = str(export_format).lower()
    if fmt == "auto":
        return "flac" if input_kind == "audio" else "h5"
    valid = {"flac", "wav", "h5", "npz"}
    if fmt not in valid:
        raise ValueError(f"Unsupported export format: {fmt!r}")
    return fmt


def _fmt_ext(export_format: str, input_kind: str) -> str:
    return {"flac": ".flac", "wav": ".wav", "h5": ".h5", "npz": ".npz"}[
        _resolve_fmt(export_format, input_kind)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# High-level: process one file
# ─────────────────────────────────────────────────────────────────────────────

def process_one_file(
    audio_path: Path,
    doe_df: pd.DataFrame,
    out_root: Path,
    expected_segments: int,
    pre_pad_s: float = 0.20,
    post_pad_s: float = 0.25,
    band_hz: tuple[float, float] = (2000.0, 5000.0),
    *,
    export_format: str = "auto",
    h5_data_key: str = "measurement/data",
    h5_time_key: str = "measurement/time_vector",
    target_sr: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Split one recording, map to DOE, export segments, return manifest rows."""
    audio_path = Path(audio_path)
    stem       = audio_path.stem
    out_dir    = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    sig = read_signal_auto(
        audio_path, target_sr=target_sr,
        h5_data_key=h5_data_key, h5_time_key=h5_time_key,
    )
    y, sr, duration_s = sig["y"], int(sig["sr"]), float(sig["duration_s"])
    tv, kind = sig["time_vector"], str(sig["input_kind"])

    segs_s, dbg     = detect_segments(y, sr, segments_per_file=expected_segments, band_hz=band_hz)
    segs_padded     = apply_padding(segs_s, pre_pad_s, post_pad_s, duration_s)
    segs_final      = refine_segment_edges(dbg, segs_padded)

    doe_mapped = map_segments_to_doe(doe_df, n_segments=len(segs_final))
    ext        = _fmt_ext(export_format, kind)

    filenames: list[str] = []
    for i, row in doe_mapped.iterrows():
        filenames.append(build_segment_filename(
            stem, int(i) + 1,
            step=row.get("Step"), hole=row.get("HoleID"), depth=row.get("Depth_mm"),
            ext=ext,
        ))

    out_paths = export_segments(
        y, sr, segs_final, out_dir, filenames,
        export_format=export_format, input_kind=kind, time_vector=tv,
        h5_data_key=h5_data_key, h5_time_key=h5_time_key,
    )

    rows: list[dict[str, Any]] = []
    for i, (core, pad, fin, p) in enumerate(
        zip(segs_s, segs_padded, segs_final, out_paths), start=1
    ):
        s0, s1 = _seg_to_samples(fin[0], fin[1], sr, len(y))
        row = doe_mapped.iloc[i - 1].to_dict()
        rows.append({
            "input_file": str(audio_path), "input_stem": stem,
            "input_kind": kind, "export_format": _resolve_fmt(export_format, kind),
            "sr_hz": int(sr), "duration_s": float(duration_s),
            "expected_segments": int(expected_segments),
            "detected_segments_core": int(len(segs_s)),
            "exported_segments_final": int(len(segs_final)),
            "split_index": int(i),
            "core_start_s": float(core[0]),  "core_end_s": float(core[1]),
            "padded_start_s": float(pad[0]), "padded_end_s": float(pad[1]),
            "start_s": float(fin[0]),        "end_s": float(fin[1]),
            "duration_seg_s": float(fin[1] - fin[0]),
            "start_sample": int(s0),         "end_sample": int(s1),
            "time_jitter_rel": float(sig.get("relative_time_jitter", 0.0)),
            "output_path": str(p.relative_to(out_root)),
            **row,
        })

    return pd.DataFrame(rows), {
        "stem": stem, "input_kind": kind, "sr_hz": int(sr),
        "expected_segments": int(expected_segments),
        "detected_segments_core": int(len(segs_s)),
        "exported_segments_final": int(len(segs_final)),
        "out_dir": str(out_dir),
    }


def process_batch(
    audio_paths: list[Path],
    doe_df: pd.DataFrame,
    out_root: Path,
    expected_map: dict[str, int],
    pre_pad_s: float = 0.20,
    post_pad_s: float = 0.25,
    band_hz: tuple[float, float] = (2000.0, 5000.0),
    *,
    export_format: str = "auto",
    h5_data_key: str = "measurement/data",
    h5_time_key: str = "measurement/time_vector",
    target_sr: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Batch-process a list of recordings; write manifest.csv; return (manifest, summary)."""
    out_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for p in audio_paths:
        p = Path(p)
        stem = p.stem
        if stem not in expected_map:
            raise KeyError(
                f"No expected segment count for {stem!r}. "
                "Add it to the expected_map or provide an expected-map CSV."
            )
        m_df, summary = process_one_file(
            p, doe_df, out_root, expected_map[stem],
            pre_pad_s=pre_pad_s, post_pad_s=post_pad_s, band_hz=band_hz,
            export_format=export_format,
            h5_data_key=h5_data_key, h5_time_key=h5_time_key,
            target_sr=target_sr,
        )
        all_rows.append(m_df)
        summaries.append(summary)

    manifest = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    manifest.to_csv(out_root / "manifest.csv", index=False)
    return manifest, pd.DataFrame(summaries)
