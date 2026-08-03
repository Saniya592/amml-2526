from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(frame: pd.DataFrame, path: Path, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(frame["class"].astype(str), frame["count"])
    ax.set_xlabel("MNIST class")
    ax.set_ylabel("Number of images")
    ax.set_title("Holdout dataset class distribution")
    for i, value in enumerate(frame["count"]):
        ax.text(i, value, str(int(value)), ha="center", va="bottom", fontsize=8)
    _save(fig, path, dpi)


def plot_reconstruction_grid(
    originals: np.ndarray,
    reconstructions_by_model: dict[str, np.ndarray],
    labels: np.ndarray,
    path: Path,
    *,
    dpi: int,
    title: str,
) -> None:
    model_names = list(reconstructions_by_model)
    rows = 1 + len(model_names)
    columns = len(originals)
    fig, axes = plt.subplots(rows, columns, figsize=(1.35 * columns, 1.45 * rows), squeeze=False)
    for column in range(columns):
        axes[0, column].imshow(originals[column].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[0, column].set_title(f"True: {int(labels[column])}", fontsize=8)
        axes[0, column].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=9)

    for row, model_name in enumerate(model_names, start=1):
        images = reconstructions_by_model[model_name]
        for column in range(columns):
            axes[row, column].imshow(images[column].squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[row, column].axis("off")
        axes[row, 0].set_ylabel(model_name, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_generated_grid(
    samples_by_model: dict[str, np.ndarray],
    path: Path,
    *,
    dpi: int,
    samples_per_model: int = 10,
) -> None:
    model_names = list(samples_by_model)
    fig, axes = plt.subplots(
        len(model_names), samples_per_model, figsize=(1.3 * samples_per_model, 1.5 * len(model_names)), squeeze=False
    )
    for row, model_name in enumerate(model_names):
        for column in range(samples_per_model):
            axes[row, column].imshow(samples_by_model[model_name][column].squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[row, column].axis("off")
        axes[row, 0].set_ylabel(model_name, fontsize=9)
    fig.suptitle("Prior samples decoded by the three supplied VAEs")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_metric_boxplot(
    frame: pd.DataFrame,
    *,
    value_column: str,
    condition_column: str,
    path: Path,
    dpi: int,
    ylabel: str,
    title: str,
) -> None:
    conditions = list(frame[condition_column].drop_duplicates())
    values = [frame.loc[frame[condition_column] == condition, value_column].dropna() for condition in conditions]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.boxplot(values, tick_labels=conditions, showfliers=False)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _save(fig, path, dpi)


def plot_confusion_matrix(
    matrix: np.ndarray,
    path: Path,
    *,
    dpi: int,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, interpolation="nearest", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, label="Recall within true class")
    ax.set(
        xticks=np.arange(10),
        yticks=np.arange(10),
        xlabel="Predicted class",
        ylabel="True class",
        title=title,
    )
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6)
    _save(fig, path, dpi)


def plot_latent_spaces(
    latent_by_model: dict[str, np.ndarray],
    labels: np.ndarray,
    path_prefix: Path,
    *,
    dpi: int,
    seed: int,
    tsne_max_samples: int,
) -> None:
    model_names = list(latent_by_model)

    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5), squeeze=False)
    for column, model_name in enumerate(model_names):
        projection = PCA(n_components=2).fit_transform(latent_by_model[model_name])
        scatter = axes[0, column].scatter(projection[:, 0], projection[:, 1], c=labels, s=8, alpha=0.65)
        axes[0, column].set_title(f"{model_name}: PCA of latent means")
        axes[0, column].set_xlabel("Principal component 1")
        axes[0, column].set_ylabel("Principal component 2")
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="MNIST class")
    _save(fig, path_prefix.with_name(path_prefix.name + "_pca.png"), dpi)

    rng = np.random.default_rng(seed)
    n = len(labels)
    selected = rng.choice(n, min(n, tsne_max_samples), replace=False)
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5), squeeze=False)
    for column, model_name in enumerate(model_names):
        z = latent_by_model[model_name][selected]
        perplexity = min(30, max(5, (len(z) - 1) // 3))
        projection = TSNE(
            n_components=2,
            random_state=seed,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        ).fit_transform(z)
        scatter = axes[0, column].scatter(projection[:, 0], projection[:, 1], c=labels[selected], s=9, alpha=0.70)
        axes[0, column].set_title(f"{model_name}: t-SNE of latent means")
        axes[0, column].set_xlabel("t-SNE dimension 1")
        axes[0, column].set_ylabel("t-SNE dimension 2")
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="MNIST class")
    _save(fig, path_prefix.with_name(path_prefix.name + "_tsne.png"), dpi)


def plot_training_history(
    history: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_total_loss"], label="Training total loss")
    ax.plot(history["epoch"], history["validation_total_loss"], label="Validation total loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean loss per sample")
    ax.set_title(title)
    ax.legend()
    _save(fig, path, dpi)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_macro_f1"], label="Training macro-F1")
    ax.plot(history["epoch"], history["validation_macro_f1"], label="Validation macro-F1")
    ax.plot(history["epoch"], history["train_balanced_accuracy"], label="Training balanced accuracy")
    ax.plot(history["epoch"], history["validation_balanced_accuracy"], label="Validation balanced accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title(title.replace("loss", "classification performance"))
    ax.legend()
    _save(fig, path.with_name(path.stem + "_classification.png"), dpi)


def plot_error_examples(
    images: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    path: Path,
    *,
    dpi: int,
    title: str,
    max_examples: int = 20,
) -> None:
    errors = np.where(labels != predictions)[0]
    if len(errors) == 0:
        return
    selected = errors[np.argsort(confidences[errors])[::-1][:max_examples]]
    columns = min(5, len(selected))
    rows = int(np.ceil(len(selected) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.3 * columns, 2.5 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, index in zip(axes.ravel(), selected, strict=False):
        axis.imshow(images[index].squeeze(), cmap="gray", vmin=0, vmax=1)
        axis.set_title(
            f"True {labels[index]} | Pred {predictions[index]}\nConfidence {confidences[index]:.2f}",
            fontsize=8,
        )
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_per_class_metric(
    frames: Iterable[pd.DataFrame],
    labels: Iterable[str],
    *,
    value_column: str,
    path: Path,
    dpi: int,
    title: str,
    ylabel: str,
) -> None:
    frames = list(frames)
    labels = list(labels)
    classes = np.arange(10)
    width = 0.8 / len(frames)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (frame, label) in enumerate(zip(frames, labels, strict=True)):
        values = frame.set_index("class").reindex(classes)[value_column].to_numpy()
        ax.bar(classes - 0.4 + width / 2 + i * width, values, width=width, label=label)
    ax.set_xticks(classes)
    ax.set_xlabel("MNIST class")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    _save(fig, path, dpi)
