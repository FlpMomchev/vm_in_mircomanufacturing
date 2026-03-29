from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .config import TrainConfig


def _safe_plot_dir(root: str | Path) -> Path:
    out_dir = Path(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_fig(fig: plt.Figure, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _regression_pred_col(df: pd.DataFrame) -> str:
    for col in ("y_pred", "y_pred_depth"):
        if col in df.columns:
            return col
    raise KeyError(
        "No regression prediction column found. Expected one of ['y_pred', 'y_pred_depth']."
    )


def save_training_overview_plots(
    file_pred_df: pd.DataFrame,
    cfg: TrainConfig,
    out_dir: str | Path,
) -> None:
    out_dir = _safe_plot_dir(out_dir)

    if cfg.task == "classification":
        save_confusion_matrix_plot(file_pred_df, out_dir / "confusion_matrix.png")
        save_per_class_metrics_plot(file_pred_df, out_dir / "per_class_metrics.png")
        save_confidence_distribution_plot(file_pred_df, out_dir / "confidence_distribution.png")
    else:
        save_regression_scatter_plot(file_pred_df, out_dir / "regression_true_vs_pred.png")
        save_error_by_depth_plot(file_pred_df, out_dir / "regression_error_by_depth.png")
        save_abs_error_hist_plot(file_pred_df, out_dir / "regression_abs_error_distribution.png")
        save_signed_error_by_depth_plot(
            file_pred_df, out_dir / "regression_signed_error_by_depth.png"
        )

    save_embedding_pca_plot(file_pred_df, out_dir / "embedding_pca.png")


def save_confusion_matrix_plot(file_pred_df: pd.DataFrame, out_path: str | Path) -> None:
    y_true = file_pred_df["y_true_class"].to_numpy()
    y_pred = file_pred_df["y_pred_class"].to_numpy()
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    im = ax.imshow(cm, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("File-level confusion matrix")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    _save_fig(fig, out_path)


def save_per_class_metrics_plot(file_pred_df: pd.DataFrame, out_path: str | Path) -> None:
    y_true = file_pred_df["y_true_class"].to_numpy()
    y_pred = file_pred_df["y_pred_class"].to_numpy()
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - width, precision, width=width, label="Precision")
    ax.bar(x, recall, width=width, label="Recall")
    ax.bar(x + width, f1, width=width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels([str(idx) for idx in labels], rotation=0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Class index")
    ax.set_ylabel("Score")
    ax.set_title("Per-class classification metrics")
    ax.legend(loc="upper center", ncol=3, frameon=False)
    _save_fig(fig, out_path)


def save_confidence_distribution_plot(file_pred_df: pd.DataFrame, out_path: str | Path) -> None:
    prob_cols = [col for col in file_pred_df.columns if col.startswith("p_")]
    if not prob_cols:
        return
    confidence = file_pred_df[prob_cols].to_numpy().max(axis=1)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.hist(confidence, bins=12)
    ax.set_xlabel("Max predicted class probability")
    ax.set_ylabel("Count")
    ax.set_title("Prediction confidence distribution")
    _save_fig(fig, out_path)


def save_regression_scatter_plot(file_pred_df: pd.DataFrame, out_path: str | Path) -> None:
    pred_col = _regression_pred_col(file_pred_df)
    y_true = file_pred_df["y_true_depth"].to_numpy()
    y_pred = file_pred_df[pred_col].to_numpy()

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(y_true, y_pred, alpha=0.85)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("True depth [mm]")
    ax.set_ylabel("Predicted depth [mm]")
    ax.set_title("Regression: true vs predicted")
    _save_fig(fig, out_path)


def save_error_by_depth_plot(file_pred_df: pd.DataFrame, out_path: str | Path) -> None:
    pred_col = _regression_pred_col(file_pred_df)
    df = file_pred_df.copy()
    df["abs_error"] = (df["y_true_depth"] - df[pred_col]).abs()
    summary = df.groupby("y_true_depth", as_index=False)["abs_error"].mean()

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(summary["y_true_depth"].astype(str), summary["abs_error"])
    ax.set_xlabel("True depth [mm]")
    ax.set_ylabel("Mean absolute error [mm]")
    ax.set_title("Regression error by depth level")
    _save_fig(fig, out_path)


def save_signed_error_by_depth_plot(file_pred_df: pd.DataFrame, out_path: str | Path) -> None:
    pred_col = _regression_pred_col(file_pred_df)
    df = file_pred_df.copy()
    df["signed_error"] = df[pred_col] - df["y_true_depth"]
    summary = df.groupby("y_true_depth", as_index=False)["signed_error"].mean()

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(summary["y_true_depth"].astype(str), summary["signed_error"])
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("True depth [mm]")
    ax.set_ylabel("Mean signed error [mm]")
    ax.set_title("Regression bias by depth level")
    _save_fig(fig, out_path)


def save_abs_error_hist_plot(file_pred_df: pd.DataFrame, out_path: str | Path) -> None:
    pred_col = _regression_pred_col(file_pred_df)
    err = np.abs(file_pred_df["y_true_depth"].to_numpy() - file_pred_df[pred_col].to_numpy())
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.hist(err, bins=16)
    ax.set_xlabel("Absolute error [mm]")
    ax.set_ylabel("Count")
    ax.set_title("Absolute error distribution")
    _save_fig(fig, out_path)


def save_embedding_pca_plot(file_pred_df: pd.DataFrame, out_path: str | Path) -> None:
    emb_cols = [col for col in file_pred_df.columns if col.startswith("emb_")]
    if len(emb_cols) < 2 or len(file_pred_df) < 3:
        return
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(file_pred_df[emb_cols].to_numpy())

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    color_values = file_pred_df["y_true_depth"].to_numpy()
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=color_values)
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="True depth [mm]")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Embedding PCA by true depth")
    _save_fig(fig, out_path)


def save_attention_maps_for_examples(
    model,
    dataset,
    file_pred_df: pd.DataFrame,
    out_dir: str | Path,
    device: str,
    n_examples: int = 4,
) -> None:
    if len(file_pred_df) == 0:
        return

    out_dir = _safe_plot_dir(out_dir)
    df = file_pred_df.copy()
    pred_col = None
    if "y_pred_class" not in df.columns:
        try:
            pred_col = _regression_pred_col(df)
        except KeyError:
            pred_col = None

    if pred_col is not None:
        df["selection_score"] = (df["y_true_depth"] - df[pred_col]).abs()
        picks = pd.concat(
            [
                df.nsmallest(max(1, n_examples // 2), "selection_score"),
                df.nlargest(max(1, n_examples // 2), "selection_score"),
            ],
            ignore_index=True,
        ).drop_duplicates(subset=["file_id"])
    else:
        prob_cols = [col for col in df.columns if col.startswith("p_")]
        if not prob_cols:
            return
        df["selection_score"] = df[prob_cols].to_numpy().max(axis=1)
        picks = pd.concat(
            [
                df.nlargest(max(1, n_examples // 2), "selection_score"),
                df.nsmallest(max(1, n_examples // 2), "selection_score"),
            ],
            ignore_index=True,
        ).drop_duplicates(subset=["file_id"])

    window_records = getattr(dataset, "window_records", None)
    if window_records is None:
        return

    file_to_indices: dict[int, list[int]] = {}
    for idx, record in enumerate(window_records):
        file_to_indices.setdefault(int(record.file_id), []).append(idx)

    model.eval()
    for _, row in picks.iterrows():
        file_id = int(row["file_id"])
        indices = file_to_indices.get(file_id, [])
        if not indices:
            continue
        mid_idx = indices[len(indices) // 2]
        batch = dataset[mid_idx]
        waveform = batch["waveform"].unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(waveform, return_extras=True)

        if "spec" not in out or "token_attention" not in out:
            continue

        spec = out["spec"].detach().cpu().squeeze().numpy()
        attn = out["token_attention"].detach().cpu().squeeze().numpy()
        h_tokens, w_tokens = out["token_grid_hw"]
        attn_2d = attn.reshape(h_tokens, w_tokens)
        attn_img = torch.tensor(attn_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        attn_img = (
            torch.nn.functional.interpolate(
                attn_img,
                size=spec.shape,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .numpy()
        )

        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        ax.imshow(spec, aspect="auto", origin="lower")
        ax.imshow(attn_img, aspect="auto", origin="lower", alpha=0.35)
        title = f"file_id={file_id} | true={row['y_true_depth']:.3f}"
        if pred_col is not None and pred_col in row.index:
            title += f" | pred={row[pred_col]:.3f}"
        ax.set_title(title)
        ax.set_xlabel("Time bins")
        ax.set_ylabel("Mel bins")
        _save_fig(fig, out_dir / f"attention_file_{file_id}.png")
