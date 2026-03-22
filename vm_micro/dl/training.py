from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR

from .config import TrainConfig
from .data import WaveformWindowDataset
from .engine import (
    aggregate_file_predictions,
    build_loss,
    evaluate_file_level,
    make_loader,
    predict_loader,
    train_one_epoch,
)
from .models import DepthModel
from .splits import build_main_split_assignments, save_split_summary, summarize_split
from .utils import dump_json, save_confusion_matrix_csv, set_seed, write_label_mapping
from .visuals import save_attention_maps_for_examples, save_training_overview_plots


@dataclass
class ExperimentResult:
    summary_df: pd.DataFrame


@dataclass
class SingleRunArtifacts:
    best_epoch: int
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]


@dataclass
class FinalModelResult:
    final_epochs: int
    final_model_dir: Path



def _build_scheduler(optimizer, cfg: TrainConfig, total_epochs: int) -> LambdaLR:
    """
    Linear warmup followed by cosine annealing to lr_min.

    During warmup (epochs 1..warmup_epochs) the LR rises linearly from
    lr_min to the base lr set in the optimizer.  After warmup it follows a
    cosine curve down to lr_min.

    The lambda receives the 0-based epoch index; we offset by 1 for readability.
    """
    warmup = max(1, cfg.warmup_epochs)
    base_lr = cfg.lr
    min_lr = cfg.lr_min

    def _lr_lambda(epoch_0based: int) -> float:
        epoch = epoch_0based + 1  # 1-based
        if epoch <= warmup:
            # linear ramp: min_lr → base_lr
            return min_lr / base_lr + (1.0 - min_lr / base_lr) * (epoch / warmup)
        # cosine decay: base_lr → min_lr
        progress = (epoch - warmup) / max(1, total_epochs - warmup)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_lr / base_lr + (1.0 - min_lr / base_lr) * cosine

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)



def _repeat_dir(root_out_dir: Path, repeat_idx: int, n_repeats: int) -> Path:
    return root_out_dir if n_repeats == 1 else root_out_dir / f"repeat_{repeat_idx:03d}"



def _make_model(cfg: TrainConfig, out_dim: int, device: str):
    model = DepthModel(cfg, out_dim=out_dim).to(device)
    if cfg.torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model



def _selection_objective(cfg: TrainConfig, metrics: dict[str, float]) -> float:
    if cfg.task == "classification":
        return float(metrics["balanced_accuracy"])
    return float(-metrics["mae"])



def _classification_labels(class_to_depth: dict[int, float]) -> list[str]:
    return [f"{class_to_depth[idx]:.3f}" for idx in sorted(class_to_depth)]



def fit_single_repeat(
    cfg: TrainConfig,
    split_df: pd.DataFrame,
    repeat_dir: Path,
    class_to_depth: dict[int, float],
    device: str,
) -> SingleRunArtifacts:
    train_df = split_df.loc[split_df["split"] == "train"].reset_index(drop=True)
    val_df = split_df.loc[split_df["split"] == "val"].reset_index(drop=True)
    test_df = split_df.loc[split_df["split"] == "test"].reset_index(drop=True)

    train_ds = WaveformWindowDataset(train_df, cfg, training=True)
    val_ds = WaveformWindowDataset(val_df, cfg, training=False)
    test_ds = WaveformWindowDataset(test_df, cfg, training=False)

    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg, shuffle=False)
    test_loader = make_loader(test_ds, cfg, shuffle=False)

    out_dim = int(split_df["class_idx"].max()) + 1 if cfg.task == "classification" else 1
    model = _make_model(cfg, out_dim, device)

    criterion = build_loss(cfg, train_df).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, cfg, cfg.epochs)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=cfg.use_amp and device == "cuda",
    )

    best_metric = None
    best_state = None
    best_epoch = 1
    bad_epochs = 0
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            cfg=cfg,
            scaler=scaler,
        )

        val_window_df = predict_loader(model, val_loader, device, cfg)
        val_file_df = aggregate_file_predictions(
            pred_df=val_window_df,
            file_df=val_df,
            cfg=cfg,
            class_to_depth=class_to_depth,
        )
        val_metrics = evaluate_file_level(val_file_df, cfg)
        objective = _selection_objective(cfg, val_metrics)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_seconds": train_stats.seconds,
                "lr": float(scheduler.get_last_lr()[0]),
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )

        print(
            f"epoch={epoch:03d} "
            f"lr={scheduler.get_last_lr()[0]:.2e} "
            f"train_loss={train_stats.loss:.4f} "
            f"val_metrics={val_metrics}"
        )

        if best_metric is None or objective > best_metric:
            best_metric = objective
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        scheduler.step()

        if bad_epochs >= cfg.patience:
            break

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(repeat_dir / "history.csv", index=False)

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), repeat_dir / "best_model.pt")

    val_window_df = predict_loader(model, val_loader, device, cfg)
    val_file_df = aggregate_file_predictions(val_window_df, val_df, cfg, class_to_depth)
    test_window_df = predict_loader(model, test_loader, device, cfg)
    test_file_df = aggregate_file_predictions(test_window_df, test_df, cfg, class_to_depth)

    val_window_df.to_csv(repeat_dir / "val_window_predictions.csv", index=False)
    val_file_df.to_csv(repeat_dir / "val_file_predictions.csv", index=False)
    test_window_df.to_csv(repeat_dir / "test_window_predictions.csv", index=False)
    test_file_df.to_csv(repeat_dir / "test_file_predictions.csv", index=False)

    val_metrics = evaluate_file_level(val_file_df, cfg)
    test_metrics = evaluate_file_level(test_file_df, cfg)

    dump_json(val_metrics, repeat_dir / "val_metrics.json")
    dump_json(test_metrics, repeat_dir / "test_metrics.json")
    dump_json(
        {
            "device": device,
            "feature_type": cfg.feature_type,
            "task": cfg.task,
            "model_type": cfg.model_type,
            "signal_num_samples": cfg.signal_num_samples(),
            "n_train_files": int(len(train_df)),
            "n_val_files": int(len(val_df)),
            "n_test_files": int(len(test_df)),
            "n_train_windows": int(len(train_ds)),
            "n_val_windows": int(len(val_ds)),
            "n_test_windows": int(len(test_ds)),
            "best_epoch": int(best_epoch),
        },
        repeat_dir / "runtime_summary.json",
    )

    if cfg.task == "classification":
        labels = _classification_labels(class_to_depth)
        save_confusion_matrix_csv(
            val_file_df["y_true_class"].to_numpy(),
            val_file_df["y_pred_class"].to_numpy(),
            labels,
            repeat_dir / "val_confusion_matrix.csv",
        )
        save_confusion_matrix_csv(
            test_file_df["y_true_class"].to_numpy(),
            test_file_df["y_pred_class"].to_numpy(),
            labels,
            repeat_dir / "test_confusion_matrix.csv",
        )

    if cfg.save_training_plots:
        save_training_overview_plots(val_file_df, cfg, repeat_dir / "plots_val")
        save_training_overview_plots(test_file_df, cfg, repeat_dir / "plots_test")

    if cfg.save_attention_maps and cfg.model_type == "hybrid_spec_transformer":
        save_attention_maps_for_examples(
            model=model,
            dataset=test_ds,
            file_pred_df=test_file_df,
            out_dir=repeat_dir / "attention_maps_test",
            device=device,
            n_examples=cfg.attention_examples,
        )

    return SingleRunArtifacts(
        best_epoch=int(best_epoch),
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )



def fit_repeated_experiment(
    cfg: TrainConfig,
    file_df: pd.DataFrame,
    root_out_dir: str | Path,
    n_repeats: int,
    split_builder,
    class_to_depth: dict[int, float],
    device: str,
) -> ExperimentResult:
    root_out_dir = Path(root_out_dir)
    root_out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []

    for repeat_idx in range(1, n_repeats + 1):
        repeat_seed = int(cfg.seed + repeat_idx - 1)
        repeat_dir = _repeat_dir(root_out_dir, repeat_idx, n_repeats)
        repeat_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n===== repeat {repeat_idx}/{n_repeats} | seed={repeat_seed} =====")
        set_seed(repeat_seed)

        split_df, split_spec = split_builder(file_df, repeat_seed)
        split_df.to_csv(repeat_dir / "split_assignments.csv", index=False)
        save_split_summary(summarize_split(split_df), repeat_dir)
        dump_json(split_spec, repeat_dir / "split_spec.json")

        artifacts = fit_single_repeat(
            cfg=cfg,
            split_df=split_df,
            repeat_dir=repeat_dir,
            class_to_depth=class_to_depth,
            device=device,
        )

        row: dict[str, object] = {
            "repeat_idx": repeat_idx,
            "seed": repeat_seed,
            "repeat_dir": str(repeat_dir),
            "best_epoch": artifacts.best_epoch,
        }
        row.update({f"val_{key}": value for key, value in artifacts.val_metrics.items()})
        row.update({f"test_{key}": value for key, value in artifacts.test_metrics.items()})
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(root_out_dir / "repeat_metrics.csv", index=False)

    if not summary_df.empty:
        numeric_cols = [
            col
            for col in summary_df.columns
            if col not in {"repeat_idx", "seed", "repeat_dir"}
        ]
        summary_payload: dict[str, float] = {}
        for col in numeric_cols:
            summary_payload[f"mean_{col}"] = float(summary_df[col].mean())
            summary_payload[f"std_{col}"] = float(summary_df[col].std(ddof=0))
        dump_json(summary_payload, root_out_dir / "repeat_metrics_summary.json")

    write_label_mapping(class_to_depth, root_out_dir / "label_mapping.json")
    return ExperimentResult(summary_df=summary_df)



def choose_final_training_epochs(
    summary_df: pd.DataFrame,
    explicit_epochs: int | None = None,
) -> int:
    if explicit_epochs is not None:
        return int(explicit_epochs)

    if summary_df.empty or "best_epoch" not in summary_df.columns:
        raise ValueError("Cannot infer final training epochs without repeat-level best_epoch data.")

    return int(np.median(summary_df["best_epoch"].to_numpy(dtype=int)))



def fit_final_model_all_files(
    cfg: TrainConfig,
    file_df: pd.DataFrame,
    root_out_dir: str | Path,
    class_to_depth: dict[int, float],
    device: str,
    final_epochs: int,
) -> FinalModelResult:
    root_out_dir = Path(root_out_dir)
    final_model_dir = root_out_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    final_seed = int(cfg.seed + 10_000)
    set_seed(final_seed)

    final_train_df = file_df.copy().reset_index(drop=True)
    final_train_df["split"] = "train"
    if "split_group_id" not in final_train_df.columns:
        final_train_df["split_group_id"] = final_train_df["file_group_id"].astype(str)

    train_ds = WaveformWindowDataset(final_train_df, cfg, training=True)
    train_loader = make_loader(train_ds, cfg, shuffle=True)

    out_dim = int(final_train_df["class_idx"].max()) + 1 if cfg.task == "classification" else 1
    model = _make_model(cfg, out_dim, device)
    criterion = build_loss(cfg, final_train_df).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, cfg, final_epochs)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=cfg.use_amp and device == "cuda",
    )

    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, int(final_epochs) + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            cfg=cfg,
            scaler=scaler,
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_seconds": train_stats.seconds,
                "lr": float(scheduler.get_last_lr()[0]),
            }
        )
        print(
            f"final_model epoch={epoch:03d}/{int(final_epochs):03d} "
            f"lr={scheduler.get_last_lr()[0]:.2e} "
            f"train_loss={train_stats.loss:.4f}"
        )
        scheduler.step()

    torch.save(model.state_dict(), final_model_dir / "best_model.pt")
    pd.DataFrame(history_rows).to_csv(final_model_dir / "history.csv", index=False)
    dump_json(cfg.to_json_dict(), final_model_dir / "config.json")
    dump_json(
        {
            "final_training_strategy": "all_files_fixed_epochs",
            "final_epochs": int(final_epochs),
            "epoch_rule": "median_best_epoch_from_repeats",
            "seed": final_seed,
            "n_train_files": int(len(final_train_df)),
            "n_train_windows": int(len(train_ds)),
            "task": cfg.task,
            "model_type": cfg.model_type,
            "feature_type": cfg.feature_type,
            "device": device,
        },
        final_model_dir / "final_model_manifest.json",
    )
    dump_json(
        {
            "device": device,
            "feature_type": cfg.feature_type,
            "task": cfg.task,
            "model_type": cfg.model_type,
            "signal_num_samples": cfg.signal_num_samples(),
            "n_train_files": int(len(final_train_df)),
            "n_train_windows": int(len(train_ds)),
            "final_epochs": int(final_epochs),
        },
        final_model_dir / "runtime_summary.json",
    )
    write_label_mapping(class_to_depth, final_model_dir / "label_mapping.json")
    final_train_df.to_csv(final_model_dir / "file_table.csv", index=False)

    return FinalModelResult(
        final_epochs=int(final_epochs),
        final_model_dir=final_model_dir,
    )



def make_main_split_builder(
    split_strategy: str,
    evaluation_unit: str,
    group_mode: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
):
    def _builder(file_df: pd.DataFrame, seed: int):
        split_df, spec = build_main_split_assignments(
            file_df=file_df,
            split_strategy=split_strategy,
            evaluation_unit=evaluation_unit,
            group_mode=group_mode,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
        return split_df, spec.to_dict()

    return _builder
