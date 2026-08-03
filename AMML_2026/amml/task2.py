from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import (
    DatasetSplit,
    IndexedTensorDataset,
    class_distribution,
    load_assignment_dataset,
    make_loader,
    save_split,
    stratified_three_way_split,
)
from .metrics import (
    classification_metrics,
    classification_report_frame,
    normalised_confusion,
    reconstruction_metrics_per_sample,
    summarise_distribution,
)
from .models import MultiTaskVAEClassifier, initialise_multitask_from_model0
from .plots import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_error_examples,
    plot_per_class_metric,
    plot_reconstruction_grid,
    plot_training_history,
)
from .reproducibility import resolve_device, seed_everything
from .statistics import aggregate_seed_metrics, paired_tests


VARIANTS = ["baseline", "weighted_ce"]
CLASSIFICATION_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "mcc",
]


@dataclass
class EpochOutput:
    total_loss: float
    reconstruction_loss: float
    kl_loss: float
    classification_loss: float
    labels: np.ndarray
    predictions: np.ndarray


def compute_class_weights(labels: torch.Tensor, train_indices: np.ndarray) -> torch.Tensor:
    selected = labels[torch.as_tensor(train_indices, dtype=torch.long)]
    counts = torch.bincount(selected, minlength=10).float()
    if torch.any(counts == 0):
        absent = torch.where(counts == 0)[0].tolist()
        raise ValueError(f"Training split is missing classes: {absent}")
    weights = selected.numel() / (10.0 * counts)
    return weights / weights.mean()


def multitask_loss(
    reconstruction: torch.Tensor,
    images: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_weights: torch.Tensor | None,
    beta_kl: float,
    reconstruction_weight: float,
    classification_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Explicit Task 2 objective with comparable mean-scaled components."""

    reconstruction_loss = F.binary_cross_entropy(reconstruction, images, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    classification_loss = F.cross_entropy(logits, labels, weight=class_weights)
    total = (
        reconstruction_weight * reconstruction_loss
        + beta_kl * kl_loss
        + classification_weight * classification_loss
    )
    return total, reconstruction_loss, kl_loss, classification_loss


def _run_epoch(
    model: MultiTaskVAEClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    class_weights: torch.Tensor | None,
    config: ExperimentConfig,
) -> EpochOutput:
    training = optimizer is not None
    model.train(training)

    total_examples = 0
    loss_sums = {"total": 0.0, "reconstruction": 0.0, "kl": 0.0, "classification": 0.0}
    all_labels, all_predictions = [], []

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            reconstruction, mu, logvar, logits = model(images)
            total, recon_loss, kl_loss, class_loss = multitask_loss(
                reconstruction,
                images,
                mu,
                logvar,
                logits,
                labels,
                class_weights=class_weights,
                beta_kl=config.beta_kl,
                reconstruction_weight=config.reconstruction_weight,
                classification_weight=config.classification_weight,
            )

            if training:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            batch_size = images.size(0)
            total_examples += batch_size
            loss_sums["total"] += float(total.detach()) * batch_size
            loss_sums["reconstruction"] += float(recon_loss.detach()) * batch_size
            loss_sums["kl"] += float(kl_loss.detach()) * batch_size
            loss_sums["classification"] += float(class_loss.detach()) * batch_size
            all_labels.append(labels.detach().cpu())
            all_predictions.append(logits.argmax(dim=1).detach().cpu())

    labels_np = torch.cat(all_labels).numpy()
    predictions_np = torch.cat(all_predictions).numpy()
    return EpochOutput(
        total_loss=loss_sums["total"] / total_examples,
        reconstruction_loss=loss_sums["reconstruction"] / total_examples,
        kl_loss=loss_sums["kl"] / total_examples,
        classification_loss=loss_sums["classification"] / total_examples,
        labels=labels_np,
        predictions=predictions_np,
    )


def train_one_model(
    model: MultiTaskVAEClassifier,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    variant: str,
    seed: int,
    class_weights: torch.Tensor | None,
    device: torch.device,
    config: ExperimentConfig,
    checkpoint_path: Path,
) -> pd.DataFrame:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history_rows = []

    for epoch in range(1, config.epochs + 1):
        train_output = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            class_weights=class_weights,
            config=config,
        )
        validation_output = _run_epoch(
            model,
            validation_loader,
            device=device,
            optimizer=None,
            class_weights=class_weights,
            config=config,
        )
        scheduler.step(validation_output.total_loss)

        train_metrics = classification_metrics(train_output.labels, train_output.predictions)
        validation_metrics = classification_metrics(
            validation_output.labels, validation_output.predictions
        )
        history_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_total_loss": train_output.total_loss,
                "train_reconstruction_loss": train_output.reconstruction_loss,
                "train_kl_loss": train_output.kl_loss,
                "train_classification_loss": train_output.classification_loss,
                "validation_total_loss": validation_output.total_loss,
                "validation_reconstruction_loss": validation_output.reconstruction_loss,
                "validation_kl_loss": validation_output.kl_loss,
                "validation_classification_loss": validation_output.classification_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
            }
        )

        if validation_output.total_loss < best_loss - 1e-5:
            best_loss = validation_output.total_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("Training ended without a valid model state.")
    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variant": variant,
            "seed": seed,
            "model_state_dict": best_state,
            "best_validation_total_loss": best_loss,
            "configuration": {
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "beta_kl": config.beta_kl,
                "reconstruction_weight": config.reconstruction_weight,
                "classification_weight": config.classification_weight,
                "dropout": config.dropout,
            },
        },
        checkpoint_path,
    )
    return pd.DataFrame(history_rows)


@torch.inference_mode()
def evaluate_multitask_model(
    model: MultiTaskVAEClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
    variant: str,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame, dict[str, np.ndarray]]:
    model.eval()
    all_images, all_reconstructions, all_labels = [], [], []
    all_predictions, all_probabilities, all_indices = [], [], []

    for images, labels, indices in loader:
        images_device = images.to(device)
        reconstruction, _, _, logits = model.deterministic_forward(images_device)
        probability = torch.softmax(logits, dim=1)
        all_images.append(images.cpu())
        all_reconstructions.append(reconstruction.cpu())
        all_labels.append(labels.cpu())
        all_predictions.append(logits.argmax(dim=1).cpu())
        all_probabilities.append(probability.cpu())
        all_indices.append(torch.as_tensor(indices).cpu())

    images = torch.cat(all_images)
    reconstructions = torch.cat(all_reconstructions)
    labels = torch.cat(all_labels).numpy()
    predictions = torch.cat(all_predictions).numpy()
    probabilities = torch.cat(all_probabilities).numpy()
    indices = torch.cat(all_indices).numpy()

    reconstruction = reconstruction_metrics_per_sample(images, reconstructions)
    class_metrics = classification_metrics(labels, predictions)
    summary = {
        "variant": variant,
        "seed": seed,
        **class_metrics,
        "reconstruction_bce": float(np.mean(reconstruction["bce"])),
        "reconstruction_mse": float(np.mean(reconstruction["mse"])),
        "reconstruction_mae": float(np.mean(reconstruction["mae"])),
        "reconstruction_ssim": float(np.mean(reconstruction["ssim"])),
    }

    per_sample = pd.DataFrame(
        {
            "variant": variant,
            "seed": seed,
            "sample_index": indices.astype(int),
            "class": labels.astype(int),
            "predicted_class": predictions.astype(int),
            "confidence": probabilities.max(axis=1),
            "correct": (labels == predictions).astype(int),
            "bce": reconstruction["bce"],
            "mse": reconstruction["mse"],
            "mae": reconstruction["mae"],
            "ssim": reconstruction["ssim"],
        }
    )
    arrays = {
        "images": images.numpy(),
        "reconstructions": reconstructions.numpy(),
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "indices": indices,
    }
    return summary, per_sample, arrays


def _load_checkpoint(
    path: Path,
    *,
    model0_path: Path,
    device: torch.device,
    dropout: float,
) -> MultiTaskVAEClassifier:
    model = initialise_multitask_from_model0(model0_path, device=device, dropout=dropout)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def _split_distribution(
    dataset: IndexedTensorDataset,
    split: DatasetSplit,
) -> pd.DataFrame:
    rows = []
    for name, indices in (
        ("train", split.train_indices),
        ("validation", split.validation_indices),
        ("test", split.test_indices),
    ):
        counts = torch.bincount(dataset.labels[torch.as_tensor(indices)], minlength=10)
        for cls, count in enumerate(counts.tolist()):
            rows.append({"split": name, "class": cls, "count": int(count)})
    return pd.DataFrame(rows)


def _per_class_results(per_sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, seed, cls), group in per_sample.groupby(["variant", "seed", "class"]):
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "class": int(cls),
                "support": int(len(group)),
                "class_recall": float(group["correct"].mean()),
                "mean_mse": float(group["mse"].mean()),
                "std_mse": float(group["mse"].std(ddof=1)) if len(group) > 1 else 0.0,
                "mean_ssim": float(group["ssim"].mean()),
                "std_ssim": float(group["ssim"].std(ddof=1)) if len(group) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _dominant_minority_classes(distribution: pd.DataFrame, number: int = 3) -> tuple[list[int], list[int]]:
    ordered = distribution.sort_values("count", ascending=False)
    dominant = ordered.head(number)["class"].astype(int).tolist()
    minority = ordered.tail(number)["class"].astype(int).tolist()
    return dominant, minority


def _group_dominance_summary(
    per_class: pd.DataFrame,
    dominant: list[int],
    minority: list[int],
) -> pd.DataFrame:
    frame = per_class.copy()
    frame["class_group"] = np.where(
        frame["class"].isin(dominant),
        "dominant",
        np.where(frame["class"].isin(minority), "minority", "middle"),
    )
    return (
        frame.groupby(["variant", "seed", "class_group"], as_index=False)
        .agg(
            mean_class_recall=("class_recall", "mean"),
            mean_reconstruction_mse=("mean_mse", "mean"),
            mean_reconstruction_ssim=("mean_ssim", "mean"),
        )
    )


def _prediction_variability(
    probability_by_variant: dict[str, list[np.ndarray]],
    labels: np.ndarray,
    indices: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for variant, probabilities in probability_by_variant.items():
        stack = np.stack(probabilities, axis=0)  # seeds x samples x classes
        mean_probability = stack.mean(axis=0)
        predicted_per_seed = stack.argmax(axis=2)
        mean_prediction = mean_probability.argmax(axis=1)
        entropy = -np.sum(mean_probability * np.log(np.clip(mean_probability, 1e-12, 1.0)), axis=1)
        probability_std = stack.std(axis=0, ddof=1).mean(axis=1)
        variation_ratio = 1.0 - np.array(
            [np.bincount(predicted_per_seed[:, i], minlength=10).max() / stack.shape[0] for i in range(stack.shape[1])]
        )
        for i in range(stack.shape[1]):
            rows.append(
                {
                    "variant": variant,
                    "sample_index": int(indices[i]),
                    "true_class": int(labels[i]),
                    "mean_predicted_class": int(mean_prediction[i]),
                    "mean_predictive_entropy": float(entropy[i]),
                    "mean_probability_std_across_seeds": float(probability_std[i]),
                    "variation_ratio": float(variation_ratio[i]),
                }
            )
    return pd.DataFrame(rows)


def _select_examples_by_classes(labels: np.ndarray, classes: list[int], per_class: int = 2) -> np.ndarray:
    selected = []
    for cls in classes:
        positions = np.where(labels == cls)[0]
        selected.extend(positions[:per_class].tolist())
    return np.asarray(selected, dtype=int)


def run_task2(config: ExperimentConfig) -> dict[str, Path]:
    """Train and compare a baseline and class-weighted small-data model."""

    config.prepare_output_directories()
    config.validate_required_files()
    device = resolve_device(config.device)
    print(f"[Task 2] Device: {device}")

    # The assignment prohibits Task 1's test dataset from Task 2. Only the
    # holdout dataset is loaded below.
    dataset = load_assignment_dataset(config.holdout_dataset_path)
    distribution = class_distribution(dataset)
    split = stratified_three_way_split(
        dataset.labels,
        test_fraction=config.test_fraction,
        validation_fraction=config.validation_fraction,
        seed=config.split_seed,
    )

    tables = config.task2_dir / "tables"
    figures = config.task2_dir / "figures"
    checkpoints = config.task2_dir / "checkpoints"
    predictions_dir = config.task2_dir / "predictions"

    distribution.to_csv(tables / "task2_holdout_class_distribution.csv", index=False)
    _split_distribution(dataset, split).to_csv(tables / "task2_split_class_distribution.csv", index=False)
    save_split(split, tables / "task2_fixed_stratified_split_indices.csv")
    plot_class_distribution(
        distribution,
        figures / "task2_holdout_class_distribution.png",
        dpi=config.save_dpi,
    )

    class_weights_cpu = compute_class_weights(dataset.labels, split.train_indices)
    pd.DataFrame(
        {
            "class": np.arange(10),
            "training_count": torch.bincount(
                dataset.labels[torch.as_tensor(split.train_indices)], minlength=10
            ).numpy(),
            "class_weight": class_weights_cpu.numpy(),
        }
    ).to_csv(tables / "task2_class_weights.csv", index=False)

    all_histories = []
    result_rows = []
    per_sample_frames = []
    probability_by_variant: dict[str, list[np.ndarray]] = {variant: [] for variant in VARIANTS}
    common_labels: np.ndarray | None = None
    common_indices: np.ndarray | None = None

    for seed in map(int, config.task2_seeds):
        for variant in VARIANTS:
            print(f"[Task 2] Training variant={variant}, seed={seed}")
            seed_everything(seed)
            train_loader = make_loader(
                dataset,
                indices=split.train_indices,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                seed=seed,
            )
            validation_loader = make_loader(
                dataset,
                indices=split.validation_indices,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                seed=seed,
            )
            test_loader = make_loader(
                dataset,
                indices=split.test_indices,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                seed=seed,
            )

            model = initialise_multitask_from_model0(
                config.model_paths["model0"],
                device=device,
                dropout=config.dropout,
            )
            weights = class_weights_cpu.to(device) if variant == "weighted_ce" else None
            checkpoint_path = checkpoints / f"task2_{variant}_seed_{seed}.pth"
            history = train_one_model(
                model,
                train_loader,
                validation_loader,
                variant=variant,
                seed=seed,
                class_weights=weights,
                device=device,
                config=config,
                checkpoint_path=checkpoint_path,
            )
            all_histories.append(history)

            summary, per_sample, arrays = evaluate_multitask_model(
                model,
                test_loader,
                device=device,
                variant=variant,
                seed=seed,
            )
            result_rows.append(summary)
            per_sample_frames.append(per_sample)
            probability_by_variant[variant].append(arrays["probabilities"])
            if common_labels is None:
                common_labels = arrays["labels"]
                common_indices = arrays["indices"]

            per_sample.to_csv(
                predictions_dir / f"task2_{variant}_seed_{seed}_test_predictions.csv",
                index=False,
            )

    histories = pd.concat(all_histories, ignore_index=True)
    results = pd.DataFrame(result_rows)
    per_sample_all = pd.concat(per_sample_frames, ignore_index=True)
    histories.to_csv(tables / "task2_training_history_all_runs.csv", index=False)
    results.to_csv(tables / "task2_test_results_by_seed.csv", index=False)
    per_sample_all.to_csv(tables / "task2_test_results_per_sample.csv", index=False)

    metric_columns = CLASSIFICATION_METRICS + [
        "reconstruction_bce",
        "reconstruction_mse",
        "reconstruction_mae",
        "reconstruction_ssim",
    ]
    aggregate_seed_metrics(
        results,
        group_columns=["variant"],
        metric_columns=metric_columns,
        confidence=config.confidence_level,
    ).to_csv(tables / "task2_test_results_summary.csv", index=False)

    paired_rows = []
    for metric in metric_columns:
        frame = paired_tests(
            results,
            condition_column="variant",
            pair_column="seed",
            value_column=metric,
            conditions=VARIANTS,
        )
        frame.insert(0, "metric", metric)
        paired_rows.append(frame)
    pd.concat(paired_rows, ignore_index=True).to_csv(
        tables / "task2_baseline_vs_weighted_paired_tests.csv", index=False
    )

    per_class = _per_class_results(per_sample_all)
    per_class.to_csv(tables / "task2_per_class_results_by_seed.csv", index=False)
    per_class_summary = (
        per_class.groupby(["variant", "class"], as_index=False)
        .agg(
            class_recall_mean=("class_recall", "mean"),
            class_recall_std=("class_recall", "std"),
            reconstruction_mse_mean=("mean_mse", "mean"),
            reconstruction_mse_std=("mean_mse", "std"),
            reconstruction_ssim_mean=("mean_ssim", "mean"),
            reconstruction_ssim_std=("mean_ssim", "std"),
        )
    )
    per_class_summary.to_csv(tables / "task2_per_class_results_summary.csv", index=False)

    dominant, minority = _dominant_minority_classes(distribution)
    dominance_summary = _group_dominance_summary(per_class, dominant, minority)
    dominance_summary.to_csv(tables / "task2_dominant_vs_minority_summary.csv", index=False)

    assert common_labels is not None and common_indices is not None
    variability = _prediction_variability(
        probability_by_variant,
        common_labels,
        common_indices,
    )
    variability.to_csv(tables / "task2_prediction_variability_across_seeds.csv", index=False)
    summarise_distribution(
        variability,
        group_columns=["variant"],
        value_columns=[
            "mean_predictive_entropy",
            "mean_probability_std_across_seeds",
            "variation_ratio",
        ],
    ).to_csv(tables / "task2_prediction_variability_summary.csv", index=False)

    # Representative run: seed closest to median macro-F1 for each variant.
    representative_arrays: dict[str, dict[str, np.ndarray]] = {}
    representative_seed: dict[str, int] = {}
    for variant in VARIANTS:
        subset = results.loc[results["variant"] == variant].copy()
        median = subset["macro_f1"].median()
        selected_row = subset.iloc[(subset["macro_f1"] - median).abs().argsort().iloc[0]]
        seed = int(selected_row["seed"])
        representative_seed[variant] = seed
        test_loader = make_loader(
            dataset,
            indices=split.test_indices,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            seed=seed,
        )
        model = _load_checkpoint(
            checkpoints / f"task2_{variant}_seed_{seed}.pth",
            model0_path=config.model_paths["model0"],
            device=device,
            dropout=config.dropout,
        )
        _, _, arrays = evaluate_multitask_model(
            model,
            test_loader,
            device=device,
            variant=variant,
            seed=seed,
        )
        representative_arrays[variant] = arrays

        matrix = normalised_confusion(arrays["labels"], arrays["predictions"])
        np.savetxt(
            tables / f"task2_{variant}_representative_normalised_confusion.csv",
            matrix,
            delimiter=",",
        )
        plot_confusion_matrix(
            matrix,
            figures / f"task2_{variant}_normalised_confusion.png",
            dpi=config.save_dpi,
            title=f"Task 2 {variant}: normalised test confusion matrix",
        )
        classification_report_frame(arrays["labels"], arrays["predictions"]).to_csv(
            tables / f"task2_{variant}_representative_classification_report.csv",
            index=False,
        )
        plot_error_examples(
            arrays["images"],
            arrays["labels"],
            arrays["predictions"],
            arrays["probabilities"].max(axis=1),
            figures / f"task2_{variant}_misclassified_examples.png",
            dpi=config.save_dpi,
            title=f"Task 2 {variant}: high-confidence test errors",
        )

        run_history = histories.loc[
            (histories["variant"] == variant) & (histories["seed"] == seed)
        ]
        plot_training_history(
            run_history,
            figures / f"task2_{variant}_training_loss.png",
            dpi=config.save_dpi,
            title=f"Task 2 {variant}: training and validation loss",
        )

    # Same examples across the baseline and mitigation model.
    labels = representative_arrays["baseline"]["labels"]
    selected = _select_examples_by_classes(labels, dominant + minority, per_class=2)
    plot_reconstruction_grid(
        representative_arrays["baseline"]["images"][selected],
        {
            "baseline": representative_arrays["baseline"]["reconstructions"][selected],
            "weighted_ce": representative_arrays["weighted_ce"]["reconstructions"][selected],
        },
        labels[selected],
        figures / "task2_dominant_and_minority_reconstructions.png",
        dpi=config.save_dpi,
        title="Task 2 reconstructions for dominant and minority classes",
    )

    # Per-class figures use averages across seeds.
    baseline_class = per_class_summary.loc[per_class_summary["variant"] == "baseline"]
    weighted_class = per_class_summary.loc[per_class_summary["variant"] == "weighted_ce"]
    plot_per_class_metric(
        [baseline_class, weighted_class],
        ["Baseline", "Weighted cross-entropy"],
        value_column="class_recall_mean",
        path=figures / "task2_per_class_recall.png",
        dpi=config.save_dpi,
        title="Task 2 mean recall by class across seeds",
        ylabel="Recall",
    )
    plot_per_class_metric(
        [baseline_class, weighted_class],
        ["Baseline", "Weighted cross-entropy"],
        value_column="reconstruction_mse_mean",
        path=figures / "task2_per_class_reconstruction_mse.png",
        dpi=config.save_dpi,
        title="Task 2 mean reconstruction MSE by class across seeds",
        ylabel="Mean squared error per pixel",
    )

    metadata: dict[str, Any] = {
        "device": str(device),
        "dataset": "holdout_dataset.pt only",
        "dataset_size": len(dataset),
        "imbalance_ratio_largest_to_smallest_nonzero_class": distribution.attrs.get("imbalance_ratio"),
        "split_seed": config.split_seed,
        "split_sizes": {
            "train": len(split.train_indices),
            "validation": len(split.validation_indices),
            "test": len(split.test_indices),
        },
        "training_seeds": list(map(int, config.task2_seeds)),
        "variants": {
            "baseline": "Model0-initialised multitask VAE with ordinary cross-entropy",
            "weighted_ce": "Identical model and split with inverse-frequency weighted cross-entropy",
        },
        "loss": {
            "total": "reconstruction_weight*BCE + beta_kl*KL + classification_weight*CE",
            "reconstruction_weight": config.reconstruction_weight,
            "beta_kl": config.beta_kl,
            "classification_weight": config.classification_weight,
        },
        "optimizer": "AdamW",
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "maximum_epochs": config.epochs,
        "early_stopping_patience": config.patience,
        "dominant_classes": dominant,
        "minority_classes": minority,
        "representative_seeds": representative_seed,
        "synthetic_data_used": False,
    }
    with (config.task2_dir / "task2_experiment_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"[Task 2] Complete. Results saved to {config.task2_dir}")
    return {
        "task2_directory": config.task2_dir,
        "results_summary": tables / "task2_test_results_summary.csv",
        "paired_tests": tables / "task2_baseline_vs_weighted_paired_tests.csv",
    }
