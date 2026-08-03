from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import IndexedTensorDataset, load_assignment_dataset, make_loader
from .metrics import (
    classification_metrics,
    classification_report_frame,
    latent_diagnostics,
    normalised_confusion,
    reconstruction_metrics_per_sample,
    summarise_distribution,
)
from .models import VariationalAutoencoder, load_pretrained_vae
from .plots import (
    plot_confusion_matrix,
    plot_error_examples,
    plot_generated_grid,
    plot_latent_spaces,
    plot_metric_boxplot,
    plot_reconstruction_grid,
)
from .reproducibility import resolve_device, seed_everything
from .statistics import aggregate_seed_metrics, friedman_test, paired_tests


MODEL_NAMES = ["model0", "model1", "model2"]
CLASSIFICATION_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "mcc",
]


def _full_loader(dataset: IndexedTensorDataset, config: ExperimentConfig, seed: int = 0) -> DataLoader:
    return make_loader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=seed,
    )


@torch.inference_mode()
def extract_latent(
    model: VariationalAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mus, logvars, labels, indices = [], [], [], []
    for images, batch_labels, batch_indices in loader:
        images = images.to(device, non_blocking=True)
        mu, logvar = model.encoder(images)
        mus.append(mu.cpu())
        logvars.append(logvar.cpu())
        labels.append(batch_labels.cpu())
        indices.append(torch.as_tensor(batch_indices).cpu())
    return (
        torch.cat(mus).numpy(),
        torch.cat(logvars).numpy(),
        torch.cat(labels).numpy(),
        torch.cat(indices).numpy(),
    )


@torch.inference_mode()
def deterministic_reconstruction_evaluation(
    model: VariationalAutoencoder,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    frames = []
    all_originals, all_reconstructions, all_labels = [], [], []
    for images, labels, indices in loader:
        images_device = images.to(device, non_blocking=True)
        mu, logvar = model.encoder(images_device)
        reconstructions = model.decoder(mu)
        metrics = reconstruction_metrics_per_sample(images_device, reconstructions)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) / mu.shape[1]
        frame = pd.DataFrame(
            {
                "model": model_name,
                "sample_index": np.asarray(indices, dtype=int),
                "class": labels.numpy().astype(int),
                "bce": metrics["bce"],
                "mse": metrics["mse"],
                "mae": metrics["mae"],
                "ssim": metrics["ssim"],
                "kl_per_latent_dimension": kl.cpu().numpy(),
                "posterior_std_mean": torch.exp(0.5 * logvar).mean(dim=1).cpu().numpy(),
            }
        )
        frames.append(frame)
        all_originals.append(images.cpu())
        all_reconstructions.append(reconstructions.cpu())
        all_labels.append(labels.cpu())

    return (
        pd.concat(frames, ignore_index=True),
        torch.cat(all_originals).numpy(),
        torch.cat(all_reconstructions).numpy(),
        torch.cat(all_labels).numpy(),
    )


@torch.inference_mode()
def stochastic_reconstruction_variability(
    model: VariationalAutoencoder,
    dataset: IndexedTensorDataset,
    *,
    device: torch.device,
    draws: int,
    max_samples: int,
    seed: int,
    model_name: str,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(dataset), min(len(dataset), max_samples), replace=False))
    loader = make_loader(
        dataset,
        indices=selected,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )
    rows = []
    for images, labels, global_indices in loader:
        images_device = images.to(device)
        mu, logvar = model.encoder(images_device)
        samples = []
        errors = []
        for _ in range(draws):
            z = model.latent_sample(mu, logvar)
            reconstruction = model.decoder(z)
            samples.append(reconstruction)
            errors.append(
                F.mse_loss(reconstruction, images_device, reduction="none").flatten(1).mean(1)
            )
        stacked = torch.stack(samples, dim=0)
        error_stack = torch.stack(errors, dim=0)
        pixel_variance = stacked.var(dim=0, unbiased=True).flatten(1).mean(1)
        mse_variance = error_stack.var(dim=0, unbiased=True)
        for i in range(len(images)):
            rows.append(
                {
                    "model": model_name,
                    "sample_index": int(global_indices[i]),
                    "class": int(labels[i]),
                    "mean_pixel_variance_across_draws": float(pixel_variance[i].cpu()),
                    "mse_variance_across_draws": float(mse_variance[i].cpu()),
                    "mean_stochastic_mse": float(error_stack[:, i].mean().cpu()),
                }
            )
    return pd.DataFrame(rows)


@torch.inference_mode()
def prior_generation_diagnostics(
    models: dict[str, VariationalAutoencoder],
    *,
    count: int,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    z_cpu = torch.randn(count, 20, generator=generator)
    rows = []
    sample_images: dict[str, np.ndarray] = {}
    for name, model in models.items():
        generated = model.decoder(z_cpu.to(device)).cpu().numpy()
        sample_images[name] = generated[:10]
        flat = generated.reshape(count, -1)
        pixel_variance = float(np.var(flat, axis=0, ddof=1).mean())
        mean_image_entropy = float(
            np.mean(
                -np.clip(flat, 1e-7, 1 - 1e-7) * np.log(np.clip(flat, 1e-7, 1 - 1e-7))
                - (1 - np.clip(flat, 1e-7, 1 - 1e-7))
                * np.log(np.clip(1 - flat, 1e-7, 1 - 1e-7))
            )
        )
        rng = np.random.default_rng(seed)
        pairs = rng.integers(0, count, size=(min(5000, count * 5), 2))
        pairwise_l2 = np.linalg.norm(flat[pairs[:, 0]] - flat[pairs[:, 1]], axis=1)
        rows.append(
            {
                "model": name,
                "mean_pixel_variance_of_prior_samples": pixel_variance,
                "mean_binary_entropy_of_prior_samples": mean_image_entropy,
                "mean_pairwise_l2_distance": float(pairwise_l2.mean()),
                "std_pairwise_l2_distance": float(pairwise_l2.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows), sample_images


def _classifier() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=3000,
                    random_state=0,
                ),
            ),
        ]
    )


def repeated_latent_classification(
    holdout_latent: dict[str, np.ndarray],
    holdout_labels: np.ndarray,
    test_latent: dict[str, np.ndarray],
    test_labels: np.ndarray,
    *,
    seeds: list[int],
    train_fraction: float,
    logistic_c: float,
    logistic_max_iter: int,
    prediction_dir: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    rows = []
    all_holdout_indices = np.arange(len(holdout_labels))
    test_predictions_by_model: dict[str, np.ndarray] = {}
    test_probabilities_by_model: dict[str, np.ndarray] = {}

    for seed in seeds:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_fraction, random_state=seed)
        train_pos, validation_pos = next(splitter.split(all_holdout_indices, holdout_labels))
        train_indices = all_holdout_indices[train_pos]
        validation_indices = all_holdout_indices[validation_pos]

        for model_name in MODEL_NAMES:
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=logistic_c,
                            solver="lbfgs",
                            max_iter=logistic_max_iter,
                            random_state=seed,
                        ),
                    ),
                ]
            )
            pipeline.fit(holdout_latent[model_name][train_indices], holdout_labels[train_indices])
            validation_predictions = pipeline.predict(holdout_latent[model_name][validation_indices])
            test_predictions = pipeline.predict(test_latent[model_name])
            test_probabilities = pipeline.predict_proba(test_latent[model_name])

            validation_metrics = classification_metrics(
                holdout_labels[validation_indices], validation_predictions
            )
            test_metrics = classification_metrics(test_labels, test_predictions)
            rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "train_size": int(len(train_indices)),
                    "validation_size": int(len(validation_indices)),
                    **{f"validation_{key}": value for key, value in validation_metrics.items()},
                    **{f"test_{key}": value for key, value in test_metrics.items()},
                }
            )

            pd.DataFrame(
                {
                    "sample_index": np.arange(len(test_labels)),
                    "true_label": test_labels,
                    "predicted_label": test_predictions,
                    "confidence": test_probabilities.max(axis=1),
                }
            ).to_csv(prediction_dir / f"{model_name}_seed_{seed}_test_predictions.csv", index=False)

    # Fit a final descriptive classifier on all holdout data. Repeated-seed
    # results remain the inferential analysis; these predictions support the
    # confusion matrices and qualitative error figures.
    for model_name in MODEL_NAMES:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=logistic_c,
                        solver="lbfgs",
                        max_iter=logistic_max_iter,
                        random_state=seeds[0],
                    ),
                ),
            ]
        )
        pipeline.fit(holdout_latent[model_name], holdout_labels)
        test_predictions_by_model[model_name] = pipeline.predict(test_latent[model_name])
        test_probabilities_by_model[model_name] = pipeline.predict_proba(test_latent[model_name])

    return pd.DataFrame(rows), test_predictions_by_model, test_probabilities_by_model


def _select_one_per_class(labels: np.ndarray) -> np.ndarray:
    selected = []
    for cls in range(10):
        positions = np.where(labels == cls)[0]
        if len(positions):
            selected.append(int(positions[0]))
    return np.asarray(selected, dtype=int)


def _select_hard_examples(reconstruction_frame: pd.DataFrame, count: int = 10) -> np.ndarray:
    average = reconstruction_frame.groupby("sample_index")["mse"].mean().sort_values(ascending=False)
    return average.head(count).index.to_numpy(dtype=int)


def _save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def run_task1(config: ExperimentConfig) -> dict[str, Path]:
    

    config.prepare_output_directories()
    config.validate_required_files()
    seed_everything(int(config.task1_seeds[0]))
    device = resolve_device(config.device)
    print(f"[Task 1] Device: {device}")

    test_dataset = load_assignment_dataset(config.test_dataset_path)
    holdout_dataset = load_assignment_dataset(config.holdout_dataset_path)
    test_loader = _full_loader(test_dataset, config)
    holdout_loader = _full_loader(holdout_dataset, config)

    models = {
        name: load_pretrained_vae(path, device)
        for name, path in config.model_paths.items()
    }

    reconstruction_frames = []
    stochastic_frames = []
    test_latent: dict[str, np.ndarray] = {}
    holdout_latent: dict[str, np.ndarray] = {}
    reconstructions_by_model: dict[str, np.ndarray] = {}
    originals: np.ndarray | None = None
    test_labels: np.ndarray | None = None
    latent_diagnostic_rows = []

    for model_name, model in models.items():
        print(f"[Task 1] Evaluating {model_name}")
        recon_frame, model_originals, model_recons, model_labels = deterministic_reconstruction_evaluation(
            model, test_loader, device, model_name
        )
        reconstruction_frames.append(recon_frame)
        reconstructions_by_model[model_name] = model_recons
        if originals is None:
            originals = model_originals
            test_labels = model_labels

        variability = stochastic_reconstruction_variability(
            model,
            test_dataset,
            device=device,
            draws=config.stochastic_recon_draws,
            max_samples=config.stochastic_recon_max_samples,
            seed=int(config.task1_seeds[0]),
            model_name=model_name,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        stochastic_frames.append(variability)

        z_test, logvar_test, y_test, _ = extract_latent(model, test_loader, device)
        z_holdout, _, y_holdout, _ = extract_latent(model, holdout_loader, device)
        test_latent[model_name] = z_test
        holdout_latent[model_name] = z_holdout
        diagnostics = latent_diagnostics(
            z_test,
            y_test,
            max_samples=config.latent_plot_max_samples,
            seed=int(config.task1_seeds[0]),
        )
        diagnostics.update(
            {
                "model": model_name,
                "mean_posterior_std": float(np.exp(0.5 * logvar_test).mean()),
                "std_posterior_std": float(np.exp(0.5 * logvar_test).std(ddof=1)),
            }
        )
        latent_diagnostic_rows.append(diagnostics)

    assert originals is not None and test_labels is not None
    holdout_labels = y_holdout
    reconstruction_frame = pd.concat(reconstruction_frames, ignore_index=True)
    stochastic_frame = pd.concat(stochastic_frames, ignore_index=True)

    tables = config.task1_dir / "tables"
    figures = config.task1_dir / "figures"
    predictions = config.task1_dir / "predictions"

    reconstruction_frame.to_csv(tables / "task1_reconstruction_per_sample.csv", index=False)
    summarise_distribution(
        reconstruction_frame,
        group_columns=["model"],
        value_columns=["bce", "mse", "mae", "ssim", "kl_per_latent_dimension", "posterior_std_mean"],
    ).to_csv(tables / "task1_reconstruction_summary.csv", index=False)
    summarise_distribution(
        reconstruction_frame,
        group_columns=["model", "class"],
        value_columns=["bce", "mse", "mae", "ssim"],
    ).to_csv(tables / "task1_reconstruction_by_class.csv", index=False)

    stochastic_frame.to_csv(tables / "task1_stochastic_variability_per_sample.csv", index=False)
    summarise_distribution(
        stochastic_frame,
        group_columns=["model"],
        value_columns=[
            "mean_pixel_variance_across_draws",
            "mse_variance_across_draws",
            "mean_stochastic_mse",
        ],
    ).to_csv(tables / "task1_stochastic_variability_summary.csv", index=False)

    # Sample-level paired tests are valid because all models process the same test images.
    reconstruction_test_rows = []
    for metric in ("bce", "mse", "mae", "ssim"):
        pairwise = paired_tests(
            reconstruction_frame,
            condition_column="model",
            pair_column="sample_index",
            value_column=metric,
            conditions=MODEL_NAMES,
        )
        pairwise.insert(0, "metric", metric)
        reconstruction_test_rows.append(pairwise)
    pd.concat(reconstruction_test_rows, ignore_index=True).to_csv(
        tables / "task1_reconstruction_pairwise_tests.csv", index=False
    )
    pd.concat(
        [
            friedman_test(
                reconstruction_frame,
                condition_column="model",
                pair_column="sample_index",
                value_column=metric,
                conditions=MODEL_NAMES,
            )
            for metric in ("bce", "mse", "mae", "ssim")
        ],
        ignore_index=True,
    ).to_csv(tables / "task1_reconstruction_friedman_tests.csv", index=False)

    pd.DataFrame(latent_diagnostic_rows).to_csv(tables / "task1_latent_diagnostics.csv", index=False)

    prior_frame, prior_images = prior_generation_diagnostics(
        models,
        count=config.prior_sample_count,
        device=device,
        seed=int(config.task1_seeds[0]),
    )
    prior_frame.to_csv(tables / "task1_prior_generation_diversity.csv", index=False)

    repeated_results, final_predictions, final_probabilities = repeated_latent_classification(
        holdout_latent,
        holdout_labels,
        test_latent,
        test_labels,
        seeds=list(map(int, config.task1_seeds)),
        train_fraction=config.task1_classifier_train_fraction,
        logistic_c=config.logistic_c,
        logistic_max_iter=config.logistic_max_iter,
        prediction_dir=predictions,
    )
    repeated_results.to_csv(tables / "task1_classification_results_by_seed.csv", index=False)
    aggregate_seed_metrics(
        repeated_results,
        group_columns=["model"],
        metric_columns=[f"test_{metric}" for metric in CLASSIFICATION_METRICS],
        confidence=config.confidence_level,
    ).to_csv(tables / "task1_classification_summary.csv", index=False)

    classification_test_rows = []
    for metric in CLASSIFICATION_METRICS:
        pairwise = paired_tests(
            repeated_results,
            condition_column="model",
            pair_column="seed",
            value_column=f"test_{metric}",
            conditions=MODEL_NAMES,
        )
        pairwise.insert(0, "metric", metric)
        classification_test_rows.append(pairwise)
    pd.concat(classification_test_rows, ignore_index=True).to_csv(
        tables / "task1_classification_pairwise_tests.csv", index=False
    )
    pd.concat(
        [
            friedman_test(
                repeated_results,
                condition_column="model",
                pair_column="seed",
                value_column=f"test_{metric}",
                conditions=MODEL_NAMES,
            ).assign(metric=metric)
            for metric in CLASSIFICATION_METRICS
        ],
        ignore_index=True,
    ).to_csv(tables / "task1_classification_friedman_tests.csv", index=False)

    # Visual comparisons use the same samples for every model.
    standard_indices = _select_one_per_class(test_labels)
    hard_indices = _select_hard_examples(reconstruction_frame, count=10)
    plot_reconstruction_grid(
        originals[standard_indices],
        {name: images[standard_indices] for name, images in reconstructions_by_model.items()},
        test_labels[standard_indices],
        figures / "task1_reconstruction_same_samples.png",
        dpi=config.save_dpi,
        title="Deterministic reconstructions of the same test examples",
    )
    plot_reconstruction_grid(
        originals[hard_indices],
        {name: images[hard_indices] for name, images in reconstructions_by_model.items()},
        test_labels[hard_indices],
        figures / "task1_reconstruction_hardest_examples.png",
        dpi=config.save_dpi,
        title="Examples with the largest average reconstruction error",
    )
    plot_generated_grid(
        prior_images,
        figures / "task1_prior_generated_samples.png",
        dpi=config.save_dpi,
    )

    for metric, label in (("mse", "Mean squared error per pixel"), ("ssim", "Structural similarity")):
        plot_metric_boxplot(
            reconstruction_frame,
            value_column=metric,
            condition_column="model",
            path=figures / f"task1_reconstruction_{metric}_boxplot.png",
            dpi=config.save_dpi,
            ylabel=label,
            title=f"Test-set reconstruction {metric.upper()} distribution",
        )

    rng = np.random.default_rng(int(config.task1_seeds[0]))
    latent_indices = np.sort(
        rng.choice(
            len(test_labels),
            min(len(test_labels), config.latent_plot_max_samples),
            replace=False,
        )
    )
    plot_latent_spaces(
        {name: values[latent_indices] for name, values in test_latent.items()},
        test_labels[latent_indices],
        figures / "task1_latent_space",
        dpi=config.save_dpi,
        seed=int(config.task1_seeds[0]),
        tsne_max_samples=config.tsne_max_samples,
    )

    for model_name in MODEL_NAMES:
        prediction = final_predictions[model_name]
        probability = final_probabilities[model_name]
        classification_report_frame(test_labels, prediction).to_csv(
            tables / f"task1_{model_name}_classification_report.csv", index=False
        )
        matrix = normalised_confusion(test_labels, prediction)
        np.savetxt(tables / f"task1_{model_name}_normalised_confusion.csv", matrix, delimiter=",")
        plot_confusion_matrix(
            matrix,
            figures / f"task1_{model_name}_normalised_confusion.png",
            dpi=config.save_dpi,
            title=f"{model_name}: normalised test confusion matrix",
        )
        plot_error_examples(
            originals,
            test_labels,
            prediction,
            probability.max(axis=1),
            figures / f"task1_{model_name}_misclassified_examples.png",
            dpi=config.save_dpi,
            title=f"{model_name}: high-confidence classification errors",
        )

    metadata = {
        "device": str(device),
        "test_size": len(test_dataset),
        "holdout_size": len(holdout_dataset),
        "models": MODEL_NAMES,
        "task1_seeds": list(map(int, config.task1_seeds)),
        "classifier": {
            "type": "StandardScaler + multinomial LogisticRegression",
            "C": config.logistic_c,
            "solver": "lbfgs",
            "max_iter": config.logistic_max_iter,
            "holdout_train_fraction": config.task1_classifier_train_fraction,
        },
        "reconstruction": "Deterministic decoder output from posterior mean mu",
        "variability": {
            "posterior_draws": config.stochastic_recon_draws,
            "maximum_samples": config.stochastic_recon_max_samples,
        },
    }
    _save_json(metadata, config.task1_dir / "task1_experiment_metadata.json")
    print(f"[Task 1] Complete. Results saved to {config.task1_dir}")
    return {
        "task1_directory": config.task1_dir,
        "classification_summary": tables / "task1_classification_summary.csv",
        "reconstruction_summary": tables / "task1_reconstruction_summary.csv",
    }
