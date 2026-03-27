from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import collate_waveforms
from .utils import classification_metrics, regression_metrics, round_to_step


@dataclass
class EpochStats:
    loss: float
    seconds: float


def make_loader(ds, cfg: TrainConfig, shuffle: bool) -> DataLoader:
    loader_kwargs: dict[str, Any] = {
        "dataset": ds,
        "batch_size": cfg.batch_size,
        "shuffle": shuffle,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "persistent_workers": cfg.persistent_workers if cfg.num_workers > 0 else False,
        "collate_fn": collate_waveforms,
    }
    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg.prefetch_factor
    return DataLoader(**loader_kwargs)


def move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_loss(cfg: TrainConfig, train_df: pd.DataFrame):
    if cfg.task == "classification":
        counts = train_df["class_idx"].value_counts().sort_index()
        max_idx = int(train_df["class_idx"].max())
        weights = torch.tensor(
            [1.0 / float(counts.get(idx, 1)) for idx in range(max_idx + 1)],
            dtype=torch.float32,
        )
        weights = weights / weights.mean()
        return torch.nn.CrossEntropyLoss(weight=weights)

    return torch.nn.HuberLoss(delta=0.1)


def _compute_loss(
    cfg: TrainConfig,
    criterion,
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    if cfg.task == "classification":
        return criterion(output["logits"], batch["y"])
    return criterion(output["regression"], batch["y"])


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device: str,
    cfg: TrainConfig,
    scaler,
) -> EpochStats:
    model.train()
    total_loss = 0.0
    n_items = 0
    started = time.perf_counter()
    use_amp = cfg.use_amp and device == "cuda"

    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=use_amp):
            output = model(batch["waveform"])
            loss = _compute_loss(cfg, criterion, output, batch)

        if use_amp:
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        batch_size = batch["waveform"].shape[0]
        total_loss += float(loss.item()) * batch_size
        n_items += batch_size

    return EpochStats(
        loss=total_loss / max(1, n_items),
        seconds=time.perf_counter() - started,
    )


@torch.no_grad()
def predict_loader(model, loader, device: str, cfg: TrainConfig) -> pd.DataFrame:
    model.eval()
    rows = []

    for batch in loader:
        paths = batch["path"]
        batch = move_batch(batch, device)
        output = model(batch["waveform"])

        if cfg.task == "classification":
            prob = output["class_probs"].cpu().numpy()
            pred_class = output["pred_class"].cpu().numpy()
            embeddings = output.get("embedding")
            embeddings_np = embeddings.cpu().numpy() if embeddings is not None else None
            file_ids = batch["file_id"].cpu().numpy()
            y_true = batch["class_idx"].cpu().numpy()
            y_depth = batch["depth_mm"].cpu().numpy()
            start_samples = batch["window_start_target"].cpu().numpy()

            for i in range(len(file_ids)):
                row = {
                    "file_id": int(file_ids[i]),
                    "window_start_target": int(start_samples[i]),
                    "y_true_class": int(y_true[i]),
                    "y_true_depth": float(y_depth[i]),
                    "y_pred_class_window": int(pred_class[i]),
                }
                for class_idx in range(prob.shape[1]):
                    row[f"p_{class_idx}"] = float(prob[i, class_idx])
                if embeddings_np is not None:
                    for emb_idx in range(embeddings_np.shape[1]):
                        row[f"emb_{emb_idx:03d}"] = float(embeddings_np[i, emb_idx])
                rows.append(row)
        else:
            pred = output["regression"].cpu().numpy()
            embeddings = output.get("embedding")
            embeddings_np = embeddings.cpu().numpy() if embeddings is not None else None
            file_ids = batch["file_id"].cpu().numpy()
            y_true = batch["depth_mm"].cpu().numpy()
            start_samples = batch["window_start_target"].cpu().numpy()

            pred = np.asarray(pred, dtype=np.float32).reshape(-1)

            if not np.isfinite(pred).all():
                bad = ~np.isfinite(pred)
                raise ValueError(
                    "Non-finite regression predictions detected. "
                    f"file_ids={file_ids[bad].tolist()} "
                    f"paths={[paths[i] for i in np.where(bad)[0]]} "
                    f"window_starts={start_samples[bad].tolist()}"
                )

            for i in range(len(file_ids)):
                row = {
                    "file_id": int(file_ids[i]),
                    "window_start_target": int(start_samples[i]),
                    "y_true_depth": float(y_true[i]),
                    "y_pred_depth_window": float(pred[i]),
                }
                if embeddings_np is not None:
                    for emb_idx in range(embeddings_np.shape[1]):
                        row[f"emb_{emb_idx:03d}"] = float(embeddings_np[i, emb_idx])
                rows.append(row)

    return pd.DataFrame(rows)


def aggregate_file_predictions(
    pred_df: pd.DataFrame,
    file_df: pd.DataFrame,
    cfg: TrainConfig,
    class_to_depth: dict[int, float] | None = None,
) -> pd.DataFrame:
    base_join_cols = [
        "file_id",
        "record_name",
        "split_group_id",
        "split",
        "depth_mm",
        "recording_root",
        "parent_dir",
        "step_idx",
    ]
    join_cols = [col for col in base_join_cols if col in file_df.columns]

    if cfg.task == "classification":
        prob_cols = [col for col in pred_df.columns if col.startswith("p_")]
        agg = pred_df.groupby("file_id")[prob_cols].mean().reset_index()
        agg["y_pred_class"] = agg[prob_cols].to_numpy().argmax(axis=1)
        if class_to_depth is None:
            raise ValueError("Classification aggregation requires class_to_depth.")
        agg["y_pred"] = agg["y_pred_class"].map(class_to_depth)

        truth = pred_df.groupby("file_id")[["y_true_class", "y_true_depth"]].first().reset_index()
        agg = agg.merge(truth, on="file_id", how="left")

        emb_cols = [col for col in pred_df.columns if col.startswith("emb_")]
        if emb_cols:
            emb_agg = pred_df.groupby("file_id")[emb_cols].mean().reset_index()
            agg = agg.merge(emb_agg, on="file_id", how="left")
        return agg.merge(file_df[join_cols], on="file_id", how="left")

    agg = (
        pred_df.groupby("file_id")["y_pred_depth_window"]
        .median()
        .reset_index()
        .rename(columns={"y_pred_depth_window": "y_pred"})
    )
    truth = pred_df.groupby("file_id")[["y_true_depth"]].first().reset_index()
    agg = agg.merge(truth, on="file_id", how="left")

    emb_cols = [col for col in pred_df.columns if col.startswith("emb_")]
    if emb_cols:
        emb_agg = pred_df.groupby("file_id")[emb_cols].mean().reset_index()
        agg = agg.merge(emb_agg, on="file_id", how="left")
    return agg.merge(file_df[join_cols], on="file_id", how="left")


def evaluate_file_level(file_pred_df: pd.DataFrame, cfg: TrainConfig) -> dict[str, float]:
    if cfg.task == "classification":
        metrics = classification_metrics(
            file_pred_df["y_true_class"].to_numpy(),
            file_pred_df["y_pred_class"].to_numpy(),
        )
        prob_cols = [col for col in file_pred_df.columns if col.startswith("p_")]
        if prob_cols:
            metrics["mean_confidence"] = float(
                file_pred_df[prob_cols].to_numpy().max(axis=1).mean()
            )
        return metrics

    metrics = regression_metrics(
        file_pred_df["y_true_depth"].to_numpy(),
        file_pred_df["y_pred"].to_numpy(),
    )
    if cfg.rounding_step_mm is not None:
        metrics["rounded_step_accuracy"] = float(
            np.mean(
                round_to_step(file_pred_df["y_true_depth"].to_numpy(), cfg.rounding_step_mm)
                == round_to_step(file_pred_df["y_pred"].to_numpy(), cfg.rounding_step_mm)
            )
        )
    metrics["mean_signed_error"] = float(
        np.mean(file_pred_df["y_pred"].to_numpy() - file_pred_df["y_true_depth"].to_numpy())
    )
    return metrics
