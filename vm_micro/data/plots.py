"""vm_micro.data.plots
~~~~~~~~~~~~~~~~~~~~~
Debug visualisations for the audio splitter.

Generates two PNGs per recording:
  <stem>__debug__core.png     detected core segments on the envelope
  <stem>__debug__padded.png   final segments with padding / ramp-up deltas annotated

Call save_debug_plots() from process_one_file() after segmentation is done.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend  safe on headless servers
import matplotlib.pyplot as plt
import numpy as np

_ADDED_COLOR = "tab:purple"
_TRIM_COLOR = "tab:red"


def save_debug_plots(
    dbg: dict[str, Any],
    segs_s: list[tuple[float, float]],
    segs_s_padded: list[tuple[float, float]] | None,
    segs_s_final: list[tuple[float, float]] | None,
    out_dir: Path,
    stem: str,
    highlight_min_ms: float = 100.0,
    dpi: int = 160,
) -> tuple[Path, Path]:
    """Save the two standard debug plots for one recording.

    Parameters
    ----------
    dbg           : Debug dict returned by detect_segments().
    segs_s        : Core detected segments (seconds).
    segs_s_padded : Segments after plain padding.
    segs_s_final  : Final segments after ramp-up edge refinement.
    out_dir       : Directory to write PNGs into.
    stem          : Recording stem used as filename prefix and plot title.
    highlight_min_ms : Minimum change (ms) to annotate with a label arrow.
    dpi           : Output image DPI.

    Returns
    -------
    (path_core_png, path_padded_png)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p_core = out_dir / f"{stem}__debug__core.png"
    p_padded = out_dir / f"{stem}__debug__padded.png"

    _plot_core(dbg, segs_s, stem, p_core, dpi)
    _plot_padded(dbg, segs_s, segs_s_padded, segs_s_final, stem, p_padded, highlight_min_ms, dpi)

    return p_core, p_padded


#
# Internal plot helpers
#


def _base_envelope_plot(dbg: dict[str, Any]) -> tuple[plt.Figure, plt.Axes]:
    """Create figure with envelope, baseline and threshold lines."""
    t = dbg["times"]
    env = dbg["env_s"]
    base = dbg["baseline"]
    thr_on = float(dbg["thr_on"])
    thr_off = float(dbg["thr_off"])

    ylo, yhi = np.percentile(env, [0.5, 99.5])

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(t, env, label="env (band, dB)", linewidth=0.8)
    ax.plot(t, base, label="baseline", linewidth=1.8)
    ax.plot(t, base + thr_on, linestyle="--", linewidth=1.0, label="baseline + thr_on")
    ax.plot(t, base + thr_off, linestyle="--", linewidth=1.0, label="baseline + thr_off")
    ax.set_ylim(ylo - 1.0, yhi + 1.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("band power (dB)")
    return fig, ax


def _plot_core(
    dbg: dict[str, Any],
    segs_s: list[tuple[float, float]],
    stem: str,
    save_path: Path,
    dpi: int,
) -> None:
    fig, ax = _base_envelope_plot(dbg)
    for a, b in segs_s:
        ax.axvspan(a, b, alpha=0.18, color="tab:blue")
    ax.set_title(f"Envelope + detected segments (core): {stem}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def _plot_padded(
    dbg: dict[str, Any],
    segs_s: list[tuple[float, float]],
    segs_s_padded: list[tuple[float, float]] | None,
    segs_s_final: list[tuple[float, float]] | None,
    stem: str,
    save_path: Path,
    highlight_min_ms: float,
    dpi: int,
) -> None:
    fig, ax = _base_envelope_plot(dbg)
    min_s = float(highlight_min_ms) / 1000.0

    # Determine what to call "final"  prefer ramp-up-refined, fall back to padded
    segs_show = segs_s_final if segs_s_final is not None else segs_s_padded

    # Gold: final segments
    if segs_show is not None:
        for a, b in segs_show:
            ax.axvspan(a, b, color="goldenrod", alpha=0.20)

    # Annotate additions/trims vs plain padding
    if (
        segs_s_padded is not None
        and segs_s_final is not None
        and len(segs_s_padded) == len(segs_s_final)
    ):
        env = np.asarray(dbg["env_s"])
        yhi = float(np.percentile(env, 99.5))
        y_anch = yhi + 0.90
        first_add = first_trim = True
        j_lab = 0

        for (a_pad, b_pad), (a_fin, b_fin) in zip(segs_s_padded, segs_s_final):
            # Earlier start added by ramp-up refinement
            ds = a_pad - a_fin
            if ds >= min_s:
                lbl = "ramp-up added" if first_add else "_nolegend_"
                ax.axvspan(a_fin, a_pad, color=_ADDED_COLOR, alpha=0.25, label=lbl)
                first_add = False
                ax.axvline(a_pad, color=_ADDED_COLOR, linestyle=":", linewidth=0.8)
                j_lab += 1
                _annotate(ax, (a_fin + a_pad) / 2, y_anch, f"+{int(round(ds * 1000))}ms", j_lab)

            # Later end added
            de = b_fin - b_pad
            if de >= min_s:
                lbl = "ramp-up added" if first_add else "_nolegend_"
                ax.axvspan(b_pad, b_fin, color=_ADDED_COLOR, alpha=0.25, label=lbl)
                first_add = False
                ax.axvline(b_pad, color=_ADDED_COLOR, linestyle=":", linewidth=0.8)
                j_lab += 1
                _annotate(ax, (b_pad + b_fin) / 2, y_anch, f"+{int(round(de * 1000))}ms", j_lab)

            # Start trimmed
            ds_t = a_fin - a_pad
            if ds_t >= min_s:
                lbl = "trimmed" if first_trim else "_nolegend_"
                ax.axvspan(a_pad, a_fin, color=_TRIM_COLOR, alpha=0.22, label=lbl)
                first_trim = False
                j_lab += 1
                _annotate(ax, (a_pad + a_fin) / 2, y_anch, f"-{int(round(ds_t * 1000))}ms", j_lab)

            # End trimmed
            de_t = b_pad - b_fin
            if de_t >= min_s:
                lbl = "trimmed" if first_trim else "_nolegend_"
                ax.axvspan(b_fin, b_pad, color=_TRIM_COLOR, alpha=0.22, label=lbl)
                first_trim = False
                j_lab += 1
                _annotate(ax, (b_fin + b_pad) / 2, y_anch, f"-{int(round(de_t * 1000))}ms", j_lab)

    # Blue: core segments on top for reference
    for a, b in segs_s:
        ax.axvspan(a, b, alpha=0.12, color="tab:blue")

    ax.set_title(f"Envelope + segments (padded / final): {stem}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def _annotate(ax: plt.Axes, x: float, y: float, text: str, j: int) -> None:
    dx = 10 if j % 2 == 0 else -12
    dy = -10 if j % 3 == 0 else -18
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left" if j % 2 == 0 else "right",
        va="top",
        fontsize=8,
        rotation=90,
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.80),
        clip_on=True,
    )
