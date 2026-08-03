from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class ExperimentConfig:
    """Central configuration for both assignment tasks.

    Paths assume this solution is copied into the root of the official
    ``pawij/amml-2526`` fork, alongside ``data/`` and ``src/``.
    """

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    output_dir_name: str = "outputs"
    device: str = "auto"
    batch_size: int = 128
    num_workers: int = 0

    # Task 1: repeated downstream classification and uncertainty analyses.
    task1_seeds: Sequence[int] = (11, 23, 37, 41, 53, 67, 79, 83, 97, 109)
    task1_classifier_train_fraction: float = 0.80
    logistic_c: float = 1.0
    logistic_max_iter: int = 3000
    latent_plot_max_samples: int = 2500
    tsne_max_samples: int = 1500
    stochastic_recon_draws: int = 20
    stochastic_recon_max_samples: int = 1000
    prior_sample_count: int = 1000

    # Task 2: fixed split plus paired training seeds.
    split_seed: int = 2026
    task2_seeds: Sequence[int] = (13, 29, 47, 71, 101)
    test_fraction: float = 0.15
    validation_fraction: float = 0.15
    epochs: int = 50
    patience: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    beta_kl: float = 0.10
    reconstruction_weight: float = 1.0
    classification_weight: float = 1.0
    dropout: float = 0.20

    # General reporting.
    confidence_level: float = 0.95
    save_dpi: int = 220

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def output_dir(self) -> Path:
        return self.project_root / self.output_dir_name

    @property
    def task1_dir(self) -> Path:
        return self.output_dir / "task1"

    @property
    def task2_dir(self) -> Path:
        return self.output_dir / "task2"

    @property
    def model_paths(self) -> dict[str, Path]:
        return {
            "model0": self.data_dir / "amml_model0_weights.pth",
            "model1": self.data_dir / "amml_model1_weights.pth",
            "model2": self.data_dir / "amml_model2_weights.pth",
        }

    @property
    def test_dataset_path(self) -> Path:
        return self.data_dir / "test_dataset.pt"

    @property
    def holdout_dataset_path(self) -> Path:
        return self.data_dir / "holdout_dataset.pt"

    def prepare_output_directories(self) -> None:
        for path in (
            self.output_dir,
            self.task1_dir,
            self.task1_dir / "figures",
            self.task1_dir / "tables",
            self.task1_dir / "predictions",
            self.task2_dir,
            self.task2_dir / "figures",
            self.task2_dir / "tables",
            self.task2_dir / "checkpoints",
            self.task2_dir / "predictions",
        ):
            path.mkdir(parents=True, exist_ok=True)

    def validate_required_files(self) -> None:
        required = [*self.model_paths.values(), self.test_dataset_path, self.holdout_dataset_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            joined = "\n  - ".join(missing)
            raise FileNotFoundError(
                "Required assignment files were not found. Copy this solution into the root "
                "of the official repository and confirm the following paths exist:\n  - " + joined
            )
