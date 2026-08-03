from __future__ import annotations

import argparse
import json
from pathlib import Path

from amml.config import ExperimentConfig
from amml.task1 import run_task1
from amml.task2 import run_task2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete AMML Summer 2026 Task 1 and Task 2 analysis."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root containing data/, src/ and this script.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0 or mps.",
    )
    parser.add_argument(
        "--task",
        choices=("all", "task1", "task2"),
        default="all",
        help="Select which assignment task to run.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduced diagnostic run. Do not use quick results in the final report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(project_root=args.project_root.resolve(), device=args.device)

    if args.quick:
        config.task1_seeds = (11, 23)
        config.task2_seeds = (13, 29)
        config.stochastic_recon_draws = 5
        config.stochastic_recon_max_samples = 250
        config.prior_sample_count = 250
        config.latent_plot_max_samples = 1000
        config.tsne_max_samples = 750
        config.epochs = 8
        config.patience = 3

    config.prepare_output_directories()
    with (config.output_dir / "resolved_configuration.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(config).items()
            },
            handle,
            indent=2,
            default=list,
        )

    if args.task in ("all", "task1"):
        run_task1(config)
    if args.task in ("all", "task2"):
        run_task2(config)


if __name__ == "__main__":
    main()
