from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from skimage.metrics import structural_similarity


@dataclass
class PredictionBundle:
    labels: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    indices: np.ndarray


def reconstruction_metrics_per_sample(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
) -> dict[str, np.ndarray]:
    originals = originals.detach().cpu().float().clamp(0, 1)
    reconstructions = reconstructions.detach().cpu().float().clamp(0, 1)

    bce = F.binary_cross_entropy(reconstructions, originals, reduction="none").flatten(1).mean(1)
    mse = F.mse_loss(reconstructions, originals, reduction="none").flatten(1).mean(1)
    mae = F.l1_loss(reconstructions, originals, reduction="none").flatten(1).mean(1)

    original_np = originals.squeeze(1).numpy()
    recon_np = reconstructions.squeeze(1).numpy()
    ssim = np.array(
        [
            structural_similarity(original_np[i], recon_np[i], data_range=1.0)
            for i in range(len(original_np))
        ],
        dtype=float,
    )
    return {
        "bce": bce.numpy(),
        "mse": mse.numpy(),
        "mae": mae.numpy(),
        "ssim": ssim,
    }


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }


def classification_report_frame(labels: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    frame = pd.DataFrame(report).T.reset_index().rename(columns={"index": "class_or_average"})
    return frame


def normalised_confusion(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return confusion_matrix(labels, predictions, labels=np.arange(10), normalize="true")


def latent_diagnostics(
    latent: np.ndarray,
    labels: np.ndarray,
    *,
    max_samples: int = 5000,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    if len(latent) > max_samples:
        selected = rng.choice(len(latent), max_samples, replace=False)
        z = latent[selected]
        y = labels[selected]
    else:
        z, y = latent, labels

    covariance = np.cov(z, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0, None)
    total_variance = float(eigenvalues.sum())
    probabilities = eigenvalues / total_variance if total_variance > 0 else np.zeros_like(eigenvalues)
    nonzero = probabilities[probabilities > 0]
    effective_rank = float(np.exp(-(nonzero * np.log(nonzero)).sum())) if len(nonzero) else 0.0

    silhouette = float(silhouette_score(z, y)) if len(np.unique(y)) > 1 else np.nan

    neighbours = NearestNeighbors(n_neighbors=2).fit(z)
    nearest = neighbours.kneighbors(return_distance=False)[:, 1]
    nn_purity = float(np.mean(y[nearest] == y))

    centroids = {cls: z[y == cls].mean(axis=0) for cls in np.unique(y)}
    within = np.mean([np.linalg.norm(row - centroids[label]) for row, label in zip(z, y, strict=True)])
    centroid_values = list(centroids.values())
    between_distances = []
    for i in range(len(centroid_values)):
        for j in range(i + 1, len(centroid_values)):
            between_distances.append(np.linalg.norm(centroid_values[i] - centroid_values[j]))
    between = float(np.mean(between_distances)) if between_distances else np.nan

    return {
        "latent_total_variance": total_variance,
        "latent_effective_rank": effective_rank,
        "silhouette": silhouette,
        "nearest_neighbour_label_purity": nn_purity,
        "mean_within_class_distance": float(within),
        "mean_between_centroid_distance": between,
        "separation_ratio": float(between / within) if within > 0 else np.nan,
    }


def summarise_distribution(
    frame: pd.DataFrame,
    group_columns: Iterable[str],
    value_columns: Iterable[str],
) -> pd.DataFrame:
    grouped = frame.groupby(list(group_columns), dropna=False)
    rows = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_columns, keys, strict=True))
        for column in value_columns:
            values = group[column].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            rows.append(
                {
                    **base,
                    "metric": column,
                    "n": int(len(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)),
                    "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)
