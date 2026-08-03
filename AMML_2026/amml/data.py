from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, TensorDataset


class IndexedTensorDataset(Dataset):
    #Tensor image dataset that returns image, class label and global index.

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        if len(images) != len(labels):
            raise ValueError("Images and labels have different lengths.")
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index], int(index)


@dataclass(frozen=True)
class DatasetSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray


def _torch_load(path: Path, *, weights_only: bool | None = None) -> Any:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if weights_only is not None:
        kwargs["weights_only"] = weights_only
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        return torch.load(path, **kwargs)


def _sample_to_image_label(sample: Any) -> tuple[Any, Any]:

    #Extract one image/label pair from a dataset sample.

    if isinstance(sample, Mapping):
        image_keys = ("images", "image", "data", "x", "X", "inputs", "features")
        label_keys = ("labels", "label", "targets", "target", "y", "Y")
        image_key = next((key for key in image_keys if key in sample), None)
        label_key = next((key for key in label_keys if key in sample), None)
        if image_key is not None and label_key is not None:
            return sample[image_key], sample[label_key]

    if isinstance(sample, Sequence) and not isinstance(sample, (str, bytes)) and len(sample) >= 2:
        return sample[0], sample[1]

    raise TypeError(
        "Each dataset item must be a mapping or a tuple/list containing at least "
        "an image and a label."
    )


def _materialise_dataset(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:

    #Materialise an arbitrary PyTorch/torchvision dataset safely.


    images: list[torch.Tensor] = []
    labels: list[int] = []

    for index in range(len(dataset)):
        image, label = _sample_to_image_label(dataset[index])

        # ``torch.as_tensor`` handles tensors and NumPy arrays.  PIL images are
        # converted through NumPy first.
        try:
            image_tensor = torch.as_tensor(image)
        except (TypeError, RuntimeError):
            image_tensor = torch.as_tensor(np.asarray(image))

        # Remove a redundant channel dimension only when materialising one
        # sample; _standardise_images adds/permutes channels consistently later.
        if image_tensor.ndim == 3 and image_tensor.shape[0] == 1:
            image_tensor = image_tensor.squeeze(0)
        elif image_tensor.ndim == 3 and image_tensor.shape[-1] == 1:
            image_tensor = image_tensor.squeeze(-1)

        images.append(image_tensor)
        labels.append(int(torch.as_tensor(label).item()))

    if not images:
        raise ValueError("The loaded dataset contains no samples.")

    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


def _extract_images_labels(obj: Any) -> tuple[Any, Any]:
    if isinstance(obj, TensorDataset):
        if len(obj.tensors) < 2:
            raise ValueError("TensorDataset contains fewer than two tensors.")
        return obj.tensors[0], obj.tensors[1]

    # The official files are commonly serialised as torch.utils.data.Subset
    # objects wrapping torchvision.datasets.MNIST.

    if isinstance(obj, Subset):
        try:
            base_images, base_labels = _extract_images_labels(obj.dataset)
            indices = torch.as_tensor(obj.indices, dtype=torch.long)
            return torch.as_tensor(base_images)[indices], torch.as_tensor(base_labels)[indices]
        except (TypeError, ValueError, IndexError, RuntimeError):
            return _materialise_dataset(obj)

    if isinstance(obj, ConcatDataset):
        return _materialise_dataset(obj)

    if isinstance(obj, Mapping):
        image_keys = ("images", "image", "data", "x", "X", "inputs", "features")
        label_keys = ("labels", "label", "targets", "target", "y", "Y")
        image_key = next((k for k in image_keys if k in obj), None)
        label_key = next((k for k in label_keys if k in obj), None)
        if image_key is not None and label_key is not None:
            return obj[image_key], obj[label_key]

        
        for key in ("dataset", "test_dataset", "holdout_dataset"):
            if key in obj:
                return _extract_images_labels(obj[key])

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)) and len(obj) >= 2:
        first, second = obj[0], obj[1]
        if torch.is_tensor(first) or isinstance(first, np.ndarray):
            return first, second

    for image_attr, label_attr in (
        ("data", "targets"),
        ("images", "labels"),
        ("x", "y"),
    ):
        if hasattr(obj, image_attr) and hasattr(obj, label_attr):
            return getattr(obj, image_attr), getattr(obj, label_attr)

    # Final compatibility path for torchvision datasets, custom Dataset
    # classes, nested wrappers and other serialised dataset containers.
    if isinstance(obj, Dataset):
        return _materialise_dataset(obj)

    raise TypeError(
        "Unsupported dataset serialization. Loaded object type: "
        f"{type(obj).__module__}.{type(obj).__qualname__}. Expected a PyTorch "
        "Dataset/Subset/TensorDataset, a tuple/list of (images, labels), or a "
        "mapping/object with data/targets or images/labels."
    )


def _standardise_images(images: Any) -> torch.Tensor:
    x = torch.as_tensor(images).detach().cpu().float()

    if x.ndim == 2 and x.shape[1] == 28 * 28:
        x = x.view(-1, 1, 28, 28)
    elif x.ndim == 3:
        x = x.unsqueeze(1)
    elif x.ndim == 4 and x.shape[-1] == 1 and x.shape[1] != 1:
        x = x.permute(0, 3, 1, 2)

    if x.ndim != 4 or x.shape[1:] != (1, 28, 28):
        raise ValueError(f"Expected MNIST images with shape [N, 1, 28, 28]; received {tuple(x.shape)}")

    finite_mask = torch.isfinite(x)
    if not finite_mask.all():
        raise ValueError("Image tensor contains NaN or infinite values.")

    if float(x.max()) > 1.5:
        x = x / 255.0
    x = x.clamp(0.0, 1.0).contiguous()
    return x


def _standardise_labels(labels: Any) -> torch.Tensor:
    y = torch.as_tensor(labels).detach().cpu().long().view(-1)
    if y.numel() == 0:
        raise ValueError("Label tensor is empty.")
    unique = torch.unique(y)
    if int(unique.min()) < 0 or int(unique.max()) > 9:
        raise ValueError(f"Expected MNIST class labels in [0, 9]; found {unique.tolist()}")
    return y.contiguous()


def load_assignment_dataset(path: Path) -> IndexedTensorDataset:
    #Load the official serialized dataset with compatibility across PyTorch versions.

    if not path.exists():
        raise FileNotFoundError(path)

    try:
        obj = _torch_load(path, weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Could not load dataset from {path}: {exc}") from exc

    images, labels = _extract_images_labels(obj)
    x = _standardise_images(images)
    y = _standardise_labels(labels)
    return IndexedTensorDataset(x, y)


def class_distribution(dataset: IndexedTensorDataset) -> pd.DataFrame:
    counts = torch.bincount(dataset.labels, minlength=10).numpy()
    total = int(counts.sum())
    rows = []
    for cls, count in enumerate(counts):
        rows.append(
            {
                "class": cls,
                "count": int(count),
                "proportion": float(count / total) if total else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    nonzero = frame.loc[frame["count"] > 0, "count"]
    if len(nonzero):
        frame.attrs["imbalance_ratio"] = float(nonzero.max() / nonzero.min())
    else:
        frame.attrs["imbalance_ratio"] = np.nan
    return frame


def stratified_three_way_split(
    labels: torch.Tensor,
    *,
    test_fraction: float,
    validation_fraction: float,
    seed: int,
) -> DatasetSplit:

    #Create a fixed stratified train/validation/test split.

     #The test set is split first, then the validation set is drawn from the remaining observations.
    

    if not (0 < test_fraction < 1 and 0 < validation_fraction < 1):
        raise ValueError("Split fractions must lie between 0 and 1.")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("Test and validation fractions must sum to less than 1.")

    y = labels.detach().cpu().numpy()
    indices = np.arange(len(y))

    first = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_val_pos, test_pos = next(first.split(indices, y))
    train_val_indices = indices[train_val_pos]
    test_indices = indices[test_pos]

    validation_within_remaining = validation_fraction / (1.0 - test_fraction)
    second = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_within_remaining,
        random_state=seed + 1,
    )
    train_pos, val_pos = next(second.split(train_val_indices, y[train_val_indices]))
    train_indices = train_val_indices[train_pos]
    validation_indices = train_val_indices[val_pos]

    return DatasetSplit(
        train_indices=np.sort(train_indices),
        validation_indices=np.sort(validation_indices),
        test_indices=np.sort(test_indices),
    )


def make_loader(
    dataset: Dataset,
    *,
    indices: np.ndarray | None = None,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    selected: Dataset = Subset(dataset, indices.tolist()) if indices is not None else dataset
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        selected,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        drop_last=False,
    )


def save_split(split: DatasetSplit, path: Path) -> None:
    rows = []
    for name, values in (
        ("train", split.train_indices),
        ("validation", split.validation_indices),
        ("test", split.test_indices),
    ):
        rows.extend({"split": name, "index": int(index)} for index in values)
    pd.DataFrame(rows).to_csv(path, index=False)
