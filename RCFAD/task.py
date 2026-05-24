"""RC-FAD: Risk-Constrained Federated Anomaly Detection.

This module keeps the Flower quickstart structure, but changes the task from
10-class image classification to binary anomaly detection.

Supported image datasets:
- cifar10
- mnist
- fmnist / fashionmnist

A chosen class is treated as the anomaly class (label=1), and all other classes
are treated as normal samples (label=0). This lets you debug the RC-FAD
algorithm before replacing the data loader with real fraud datasets.
"""

from __future__ import annotations

import os
import random
import urllib.request
import zipfile

# Set deterministic/CUDA-related environment variables before torch is used.
os.environ.setdefault("RAY_STARTUP_TIMEOUT_S", "300")
os.environ.setdefault("RAY_DISABLE_BOOTSTRAP", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor

# Ray/Flower simulation stability settings are configured above before torch use.

TABULAR_INPUT_DIM = 600


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set Python/NumPy/PyTorch seeds for reproducible FL simulations.

    Flower/Ray simulation can still introduce small nondeterminism on GPU, but
    this fixes the dominant sources: server model initialization, client-side
    shuffling, and dataset partitioning.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


@dataclass
class TrainProcessMetadata:
    """Extra information printed by CustomFedAdagrad."""

    training_time: float
    converged: bool
    training_losses: Dict[str, float]
    risk_metrics: Dict[str, float] = field(default_factory=dict)


class Net(nn.Module):
    """Small binary anomaly detector for image and tabular datasets.

    For MNIST/Fashion-MNIST, grayscale images are converted to 3-channel 32x32
    tensors, so the same model works for CIFAR-10 and MNIST-like datasets.
    Public tabular anomaly datasets use a padded/truncated vector branch, while
    MLP branch. Both branches expose 84-dimensional features for contrastive
    baselines and one binary anomaly logit.
    """

    def __init__(self):
        super().__init__()
        self.tab_fc1 = nn.Linear(TABULAR_INPUT_DIM, 64)
        self.tab_fc2 = nn.Linear(64, 84)
        self.tab_fc3 = nn.Linear(84, 1)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # Binary anomaly score logit. Use sigmoid(logit) as anomaly probability.
        self.fc3 = nn.Linear(84, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = F.relu(self.tab_fc1(x))
            x = F.relu(self.tab_fc2(x))
            return x
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        if x.dim() == 2:
            return self.tab_fc3(features).view(-1)
        return self.fc3(features).view(-1)


class BinaryAnomalyDataset(Dataset):
    """Map a multi-class torchvision dataset to binary anomaly labels."""

    def __init__(self, base_dataset: Dataset, anomaly_class: int):
        self.base_dataset = base_dataset
        self.anomaly_class = int(anomaly_class)

        if hasattr(base_dataset, "targets"):
            self.targets = np.asarray(base_dataset.targets)
        elif hasattr(base_dataset, "labels"):
            self.targets = np.asarray(base_dataset.labels)
        else:
            # Fallback: materialize labels once
            self.targets = np.asarray([base_dataset[i][1] for i in range(len(base_dataset))])

        self.binary_targets = (self.targets == self.anomaly_class).astype(np.float32)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img, original_label = self.base_dataset[idx]
        label = 1.0 if int(original_label) == self.anomaly_class else 0.0
        return {
            "img": img,
            "label": torch.tensor(label, dtype=torch.float32),
        }


class NormalScoreShiftDataset(Dataset):
    """Apply deterministic client-specific noise to normal samples.

    This creates heterogeneous normal-score distributions while leaving
    anomaly labels untouched, which is useful for threshold-personalization
    experiments.
    """

    def __init__(self, base_dataset: Dataset, noise_std: float, seed: int):
        self.base_dataset = base_dataset
        self.noise_std = float(noise_std)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base_dataset[idx]
        img = item["img"]
        label = item["label"]
        if self.noise_std > 0.0 and float(label.item()) <= 0.5:
            generator = torch.Generator().manual_seed(self.seed + int(idx))
            noise = torch.randn(img.shape, generator=generator, dtype=img.dtype) * self.noise_std
            img = torch.clamp(img + noise, -1.0, 1.0)
        return {"img": img, "label": label}


class TabularAnomalyDataset(Dataset):
    """Dataset wrapper using the existing ``img`` key for tabular features."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        features = features.astype(np.float32, copy=False)
        if features.ndim != 2:
            raise ValueError(f"Tabular features must be 2D, got shape={features.shape}.")
        self.raw_feature_dim = int(features.shape[1])
        # Keep the raw matrix compact. Padding an entire large fraud dataset to
        # the maximum model width would multiply memory across Ray client actors.
        self.features = features.astype(np.float32, copy=False)
        self.binary_targets = labels.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(len(self.binary_targets))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        feature = self.features[idx]
        if self.raw_feature_dim < TABULAR_INPUT_DIM:
            padded = np.zeros(TABULAR_INPUT_DIM, dtype=np.float32)
            padded[: self.raw_feature_dim] = feature
            feature = padded
        elif self.raw_feature_dim > TABULAR_INPUT_DIM:
            feature = feature[:TABULAR_INPUT_DIM]
        return {
            "img": torch.from_numpy(feature),
            "label": torch.tensor(self.binary_targets[idx], dtype=torch.float32),
        }


def _dataset_transforms(dataset_name: str) -> Compose:
    dataset_name = dataset_name.lower()
    if dataset_name in {"mnist", "fmnist", "fashionmnist", "fashion-mnist"}:
        # Convert 1-channel 28x28 to 3-channel 32x32 to keep one universal Net.
        return Compose([
            Resize((32, 32)),
            Grayscale(num_output_channels=3),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    return Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


@lru_cache(maxsize=16)
def _load_base_dataset(dataset_name: str, train: bool, data_root: str) -> Dataset:
    """Lazy dataset loader.

    The function is cached so repeated ClientApp calls do not reload the same
    torchvision dataset again and again in the same process.
    """

    dataset_name = dataset_name.lower()
    transform = _dataset_transforms(dataset_name)

    if dataset_name == "cifar10":
        return torchvision.datasets.CIFAR10(
            root=data_root, train=train, download=True, transform=transform
        )
    if dataset_name == "mnist":
        return torchvision.datasets.MNIST(
            root=data_root, train=train, download=True, transform=transform
        )
    if dataset_name in {"fmnist", "fashionmnist", "fashion-mnist"}:
        return torchvision.datasets.FashionMNIST(
            root=data_root, train=train, download=True, transform=transform
        )

    raise ValueError(
        f"Unsupported dataset_name={dataset_name!r}. "
        "Use cifar10, mnist, fmnist, creditcard, baf, ai4i, or secom."
    )


def _is_creditcard_dataset(dataset_name: str) -> bool:
    name = dataset_name.lower().replace("-", "").replace("_", "")
    return name in {"creditcard", "creditcardfraud", "creditfraud"}


def _normalize_dataset_key(dataset_name: str) -> str:
    name = dataset_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "creditcard": "creditcard",
        "creditcardfraud": "creditcard",
        "creditfraud": "creditcard",
        "baf": "baf",
        "bankaccountfraud": "baf",
        "bankfraud": "baf",
        "ai4i": "ai4i",
        "ai4i2020": "ai4i",
        "predictivemaintenance": "ai4i",
        "secom": "secom",
    }
    return aliases.get(name, name)


def _is_tabular_dataset(dataset_name: str) -> bool:
    return _normalize_dataset_key(dataset_name) in {"creditcard", "baf", "ai4i", "secom"}


def _download_and_extract_zip(url: str, zip_path: str, extract_dir: str) -> None:
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    print(f"Downloading {url} to {zip_path} ...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)


def _first_existing_path(paths: Iterable[str]) -> str | None:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def _find_first_csv(root: str, keywords: Iterable[str]) -> str | None:
    keys = [key.lower() for key in keywords]
    if not os.path.exists(root):
        return None
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            lower = filename.lower()
            if lower.endswith(".csv") and all(key in lower for key in keys):
                return os.path.join(dirpath, filename)
    return None


def _ensure_creditcard_csv(data_root: str) -> str:
    root = os.path.abspath(data_root)
    os.makedirs(root, exist_ok=True)
    candidates = [
        os.path.join(root, "creditcard.csv"),
        os.path.join(root, "creditcard", "creditcard.csv"),
        os.path.join(root, "creditcardfraud", "creditcard.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    csv_path = os.path.join(root, "creditcard.csv")
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    try:
        print(f"Downloading Credit Card Fraud dataset to {csv_path} ...")
        urllib.request.urlretrieve(url, csv_path)
    except Exception as exc:
        raise FileNotFoundError(
            "Credit Card Fraud dataset not found. Put Kaggle's creditcard.csv "
            f"under {root}/creditcard.csv or {root}/creditcard/creditcard.csv. "
            f"Automatic download failed: {exc}"
        ) from exc

    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Downloaded Credit Card CSV, but creditcard.csv was not found."
    )


def _ensure_baf_csv(data_root: str) -> str:
    """Return a local Bank Account Fraud CSV.

    BAF is distributed through Kaggle/NeurIPS and usually requires manual
    download.  The loader accepts the official Base.csv or any variant CSV with
    a fraud_bool target column.
    """

    root = os.path.abspath(data_root)
    candidates = [
        os.path.join(root, "baf", "Base.csv"),
        os.path.join(root, "baf", "base.csv"),
        os.path.join(root, "baf", "BAF.csv"),
        os.path.join(root, "baf", "baf.csv"),
        os.path.join(root, "bank_account_fraud", "Base.csv"),
        os.path.join(root, "Base.csv"),
        os.path.join(root, "baf.csv"),
    ]
    found = _first_existing_path(candidates)
    if found is not None:
        return found
    found = _find_first_csv(os.path.join(root, "baf"), [])
    if found is not None:
        return found
    raise FileNotFoundError(
        "BAF dataset not found. Download the Bank Account Fraud dataset "
        "from Kaggle/NeurIPS and place Base.csv under data/baf/Base.csv "
        "or pass --extra \"data-root=/path/to/data\"."
    )


def _ensure_ai4i_csv(data_root: str) -> str:
    root = os.path.abspath(data_root)
    candidates = [
        os.path.join(root, "ai4i", "AI4I 2020 Predictive Maintenance Dataset.csv"),
        os.path.join(root, "ai4i", "ai4i2020.csv"),
        os.path.join(root, "ai4i.csv"),
        os.path.join(root, "ai4i2020.csv"),
    ]
    found = _first_existing_path(candidates)
    if found is not None:
        return found
    extract_dir = os.path.join(root, "ai4i")
    zip_path = os.path.join(root, "ai4i.zip")
    url = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
    try:
        _download_and_extract_zip(url, zip_path, extract_dir)
    except Exception as exc:
        raise FileNotFoundError(
            "AI4I 2020 dataset not found and automatic UCI download failed. "
            f"Put the CSV under {extract_dir}. Error: {exc}"
        ) from exc
    found = _first_existing_path(candidates) or _find_first_csv(extract_dir, ["ai4i"])
    if found is None:
        raise FileNotFoundError(f"AI4I archive extracted, but no CSV was found under {extract_dir}.")
    return found


def _ensure_secom_files(data_root: str) -> Tuple[str, str]:
    root = os.path.abspath(data_root)
    candidates = [
        (
            os.path.join(root, "secom", "secom.data"),
            os.path.join(root, "secom", "secom_labels.data"),
        ),
        (
            os.path.join(root, "secom.data"),
            os.path.join(root, "secom_labels.data"),
        ),
    ]
    for data_path, label_path in candidates:
        if os.path.exists(data_path) and os.path.exists(label_path):
            return data_path, label_path
    extract_dir = os.path.join(root, "secom")
    zip_path = os.path.join(root, "secom.zip")
    url = "https://archive.ics.uci.edu/static/public/179/secom.zip"
    try:
        _download_and_extract_zip(url, zip_path, extract_dir)
    except Exception as exc:
        raise FileNotFoundError(
            "SECOM dataset not found and automatic UCI download failed. "
            f"Put secom.data and secom_labels.data under {extract_dir}. Error: {exc}"
        ) from exc
    for data_path, label_path in candidates:
        if os.path.exists(data_path) and os.path.exists(label_path):
            return data_path, label_path
    raise FileNotFoundError(f"SECOM archive extracted, but expected files were not found under {extract_dir}.")


def _dataframe_to_features(
    df: pd.DataFrame,
    *,
    label_col: str,
    positive_values: Iterable[Any] = (1,),
    drop_cols: Iterable[str] = (),
) -> Tuple[np.ndarray, np.ndarray]:
    if label_col not in df.columns:
        raise ValueError(f"Target column {label_col!r} not found. Columns: {list(df.columns)[:20]}")

    positive = {str(value).lower() for value in positive_values}
    labels = df[label_col].map(lambda value: 1.0 if str(value).lower() in positive else 0.0).to_numpy(dtype=np.float32)

    drops = {label_col}
    lower_to_col = {col.lower(): col for col in df.columns}
    for col in drop_cols:
        if col in df.columns:
            drops.add(col)
        elif col.lower() in lower_to_col:
            drops.add(lower_to_col[col.lower()])

    feature_df = df.drop(columns=[col for col in drops if col in df.columns]).copy()
    for col in feature_df.columns:
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
        else:
            feature_df[col] = feature_df[col].astype("string").fillna("missing")
    feature_df = pd.get_dummies(feature_df, dummy_na=False)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    medians = feature_df.median(numeric_only=True)
    feature_df = feature_df.fillna(medians).fillna(0.0)
    return feature_df.astype(np.float32).to_numpy(), labels


def _split_standardize_tabular(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = labels.astype(np.float32, copy=False)
    if int((labels == 1).sum()) == 0 or int((labels == 0).sum()) == 0:
        raise ValueError("Tabular anomaly dataset must contain both normal and anomaly labels.")
    rng = np.random.default_rng(int(seed))
    pos_idx = rng.permutation(np.where(labels == 1)[0])
    neg_idx = rng.permutation(np.where(labels == 0)[0])

    def split_indices(indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        test_size = max(1, int(round(0.2 * len(indices))))
        return indices[test_size:], indices[:test_size]

    pos_train, pos_test = split_indices(pos_idx)
    neg_train, neg_test = split_indices(neg_idx)
    train_idx = rng.permutation(np.concatenate([pos_train, neg_train]))
    test_idx = rng.permutation(np.concatenate([pos_test, neg_test]))

    x_train = features[train_idx].copy()
    y_train = labels[train_idx].copy()
    x_test = features[test_idx].copy()
    y_test = labels[test_idx].copy()

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    return x_train.astype(np.float32), y_train, x_test.astype(np.float32), y_test


@lru_cache(maxsize=16)
def _load_tabular_arrays(
    dataset_name: str,
    data_root: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load, encode, split, and normalize supported tabular anomaly datasets."""

    key = _normalize_dataset_key(dataset_name)
    if key == "creditcard":
        csv_path = _ensure_creditcard_csv(data_root)
        df = pd.read_csv(csv_path)
        features, labels = _dataframe_to_features(df, label_col="Class", positive_values=(1,))
    elif key == "baf":
        csv_path = _ensure_baf_csv(data_root)
        df = pd.read_csv(csv_path)
        features, labels = _dataframe_to_features(
            df,
            label_col="fraud_bool",
            positive_values=(1, True, "true"),
            drop_cols=("device_fraud_count",),
        )
    elif key == "ai4i":
        csv_path = _ensure_ai4i_csv(data_root)
        df = pd.read_csv(csv_path)
        features, labels = _dataframe_to_features(
            df,
            label_col="Machine failure",
            positive_values=(1,),
            drop_cols=("UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"),
        )
    elif key == "secom":
        data_path, label_path = _ensure_secom_files(data_root)
        feature_df = pd.read_csv(data_path, sep=r"\s+", header=None, na_values=["NaN"])
        label_df = pd.read_csv(label_path, sep=r"\s+", header=None)
        labels = (label_df.iloc[:, 0].astype(int).to_numpy() == 1).astype(np.float32)
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.fillna(feature_df.median(numeric_only=True)).fillna(0.0)
        features = feature_df.astype(np.float32).to_numpy()
    else:
        raise ValueError(f"Unsupported tabular dataset {dataset_name!r}.")

    return _split_standardize_tabular(features, labels, int(seed))


def _load_tabular_dataset(data_root: str, dataset_name: str, seed: int, train: bool) -> TabularAnomalyDataset:
    x_train, y_train, x_test, y_test = _load_tabular_arrays(dataset_name, data_root, int(seed))
    if train:
        return TabularAnomalyDataset(x_train, y_train)
    return TabularAnomalyDataset(x_test, y_test)


def _build_client_indices(
    binary_targets: np.ndarray,
    partition_id: int,
    num_partitions: int,
    scheme: str,
    seed: int,
    min_anomaly_ratio: float,
    max_anomaly_ratio: float,
    dirichlet_alpha: float = 0.3,
) -> np.ndarray:
    """Create client indices for IID or risk-heterogeneous partitions."""

    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(binary_targets))
    scheme = scheme.lower()

    if scheme == "iid":
        shuffled = rng.permutation(all_indices)
        return np.array_split(shuffled, num_partitions)[partition_id]

    pos_indices = rng.permutation(np.where(binary_targets == 1)[0])
    neg_indices = rng.permutation(np.where(binary_targets == 0)[0])

    if scheme in {"risk_dirichlet", "risk-dirichlet", "dirichlet"}:
        alpha = max(float(dirichlet_alpha), 1e-3)

        def split_by_dirichlet(indices: np.ndarray) -> List[np.ndarray]:
            if len(indices) == 0:
                return [np.array([], dtype=int) for _ in range(num_partitions)]
            probs = rng.dirichlet(np.full(num_partitions, alpha, dtype=float))
            counts = rng.multinomial(len(indices), probs)
            offsets = np.cumsum(counts)[:-1]
            return [part.astype(int) for part in np.split(indices, offsets)]

        pos_splits = split_by_dirichlet(pos_indices)
        neg_splits = split_by_dirichlet(neg_indices)
        client_indices = np.concatenate([pos_splits[partition_id], neg_splits[partition_id]])
        if len(client_indices) < 2:
            extra = rng.choice(all_indices, size=2 - len(client_indices), replace=True)
            client_indices = np.concatenate([client_indices, extra])
        return rng.permutation(client_indices)

    if scheme not in {"risk_hetero", "risk-hetero", "hetero"}:
        raise ValueError(f"Unknown partition_scheme={scheme!r}")

    # Same nominal client size for all clients, but different positive ratios.
    client_size = int(len(binary_targets) / num_partitions)
    ratios = np.linspace(min_anomaly_ratio, max_anomaly_ratio, num_partitions)
    ratios = rng.permutation(ratios)

    all_client_indices: List[np.ndarray] = []
    pos_cursor, neg_cursor = 0, 0

    for client_id in range(num_partitions):
        ratio = float(ratios[client_id])
        pos_count = int(round(client_size * ratio))
        pos_count = max(1, pos_count) if len(pos_indices) > 0 else 0
        neg_count = max(1, client_size - pos_count)

        # Take positives; if insufficient, sample remaining with replacement.
        if pos_cursor + pos_count <= len(pos_indices):
            pos_take = pos_indices[pos_cursor: pos_cursor + pos_count]
            pos_cursor += pos_count
        else:
            remaining = pos_indices[pos_cursor:]
            need = pos_count - len(remaining)
            extra = rng.choice(pos_indices, size=need, replace=True) if len(pos_indices) > 0 else np.array([], dtype=int)
            pos_take = np.concatenate([remaining, extra])
            pos_cursor = len(pos_indices)

        # Take negatives; if insufficient, sample remaining with replacement.
        if neg_cursor + neg_count <= len(neg_indices):
            neg_take = neg_indices[neg_cursor: neg_cursor + neg_count]
            neg_cursor += neg_count
        else:
            remaining = neg_indices[neg_cursor:]
            need = neg_count - len(remaining)
            extra = rng.choice(neg_indices, size=need, replace=True) if len(neg_indices) > 0 else np.array([], dtype=int)
            neg_take = np.concatenate([remaining, extra])
            neg_cursor = len(neg_indices)

        client_indices = rng.permutation(np.concatenate([pos_take, neg_take]))
        all_client_indices.append(client_indices)

    return all_client_indices[partition_id]


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    dataset_name: str = "mnist",
    anomaly_class: int = 1,
    partition_scheme: str = "risk_hetero",
    seed: int = 42,
    min_anomaly_ratio: float = 0.005,
    max_anomaly_ratio: float = 0.10,
    dirichlet_alpha: float = 0.3,
    data_root: str = "./data",
    dataloader_seed: int | None = None,
    normal_shift_max: float = 0.0,
) -> Tuple[DataLoader, DataLoader]:
    """Load one client's train/validation data."""

    if _is_tabular_dataset(dataset_name):
        binary_train = _load_tabular_dataset(
            data_root=data_root,
            dataset_name=dataset_name,
            seed=seed,
            train=True,
        )
    else:
        base_train = _load_base_dataset(dataset_name, train=True, data_root=data_root)
        binary_train = BinaryAnomalyDataset(base_train, anomaly_class=anomaly_class)

    indices = _build_client_indices(
        binary_targets=binary_train.binary_targets,
        partition_id=partition_id,
        num_partitions=num_partitions,
        scheme=partition_scheme,
        seed=seed,
        min_anomaly_ratio=min_anomaly_ratio,
        max_anomaly_ratio=max_anomaly_ratio,
        dirichlet_alpha=dirichlet_alpha,
    )

    client_dataset: Dataset = Subset(binary_train, indices.tolist())
    if float(normal_shift_max) > 0.0:
        if num_partitions <= 1:
            noise_std = float(normal_shift_max)
        else:
            noise_std = float(normal_shift_max) * (partition_id / max(1, num_partitions - 1))
        client_dataset = NormalScoreShiftDataset(
            client_dataset,
            noise_std=noise_std,
            seed=seed + partition_id * 9973,
        )

    val_size = max(1, int(0.2 * len(client_dataset)))
    train_size = len(client_dataset) - val_size
    if train_size <= 0:
        raise ValueError("Client partition too small. Reduce num_partitions.")

    split_generator = torch.Generator().manual_seed(seed + partition_id)
    train_subset, val_subset = random_split(
        client_dataset,
        [train_size, val_size],
        generator=split_generator,
    )

    # Fix the local shuffle order. Use a different seed per round when callers
    # pass dataloader_seed, otherwise use a stable client-specific seed.
    loader_seed = int(dataloader_seed if dataloader_seed is not None else seed + partition_id * 1000)
    train_generator = torch.Generator().manual_seed(loader_seed)

    trainloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
    )
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    return trainloader, valloader


def load_centralized_dataset(
    dataset_name: str = "mnist",
    anomaly_class: int = 1,
    batch_size: int = 128,
    data_root: str = "./data",
    seed: int = 42,
) -> DataLoader:
    """Load centralized test set for server-side evaluation."""

    if _is_tabular_dataset(dataset_name):
        binary_test = _load_tabular_dataset(
            data_root=data_root,
            dataset_name=dataset_name,
            seed=seed,
            train=False,
        )
    else:
        base_test = _load_base_dataset(dataset_name, train=False, data_root=data_root)
        binary_test = BinaryAnomalyDataset(base_test, anomaly_class=anomaly_class)
    return DataLoader(binary_test, batch_size=batch_size, shuffle=False, num_workers=0)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _compute_auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = float((labels == 1).sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    precision = tp / (np.arange(len(labels)) + 1)
    recall = tp / n_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def _recall_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    pos_total = float((labels == 1).sum())
    neg_total = float((labels == 0).sum())
    if pos_total == 0 or neg_total == 0:
        return 0.0
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    recall = tp / pos_total
    fpr = fp / neg_total
    valid = fpr <= target_fpr
    if not np.any(valid):
        return 0.0
    return float(np.max(recall[valid]))


def _threshold_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    """Return a score threshold whose negative pass rate is at most target_fpr."""
    neg_scores = scores[labels == 0]
    if len(neg_scores) == 0:
        return 0.5
    # Predict anomaly when score >= threshold. A plain quantile can collapse to
    # 0.0 when many scores are tied, which then predicts every negative sample
    # as anomalous. Step through score cutoffs and keep the best feasible one.
    target = float(np.clip(target_fpr, 0.0, 1.0))
    candidates = np.unique(neg_scores)
    best_threshold = _next_score_after_max(neg_scores)
    best_fpr = 0.0
    for threshold in candidates:
        fpr = float(np.mean(neg_scores >= threshold))
        if fpr <= target and fpr >= best_fpr:
            best_threshold = float(threshold)
            best_fpr = fpr
    return best_threshold


def _next_score_after_max(scores: np.ndarray) -> float:
    arr = np.asarray(scores)
    max_score = np.max(arr)
    if np.issubdtype(arr.dtype, np.floating):
        return float(
            np.nextafter(
                np.asarray(max_score, dtype=arr.dtype),
                np.asarray(np.inf, dtype=arr.dtype),
            )
        )
    return float(max_score) + 1.0


def _basic_hard_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(np.int64)
    y = labels.astype(np.int64)

    tp = float(((preds == 1) & (y == 1)).sum())
    fp = float(((preds == 1) & (y == 0)).sum())
    tn = float(((preds == 0) & (y == 0)).sum())
    fn = float(((preds == 0) & (y == 1)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    acc = _safe_div(tp + tn, tp + fp + tn + fn)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "fnr": fnr,
        "f1": f1,
    }


def _hard_binary_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    target_fpr: float,
) -> Dict[str, float]:
    """Compute fixed-threshold and target-FPR-calibrated metrics."""
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    fixed = _basic_hard_metrics(probs, labels, threshold)
    auc = _compute_auc(probs, labels)
    auprc = _compute_auprc(probs, labels)
    recall_at_fpr = _recall_at_fpr(probs, labels, target_fpr)

    thr_at_fpr = _threshold_at_fpr(probs, labels, target_fpr)
    calibrated = _basic_hard_metrics(probs, labels, thr_at_fpr)
    if calibrated["fpr"] > float(target_fpr) + 1e-12:
        neg_scores = probs[labels == 0]
        if len(neg_scores) > 0:
            thr_at_fpr = _next_score_after_max(neg_scores)
            calibrated = _basic_hard_metrics(probs, labels, thr_at_fpr)

    out: Dict[str, float] = {
        **fixed,
        "auc": auc,
        "auprc": auprc,
        # Ranking metric: maximum recall achievable under target FPR.
        "recall_at_fpr": recall_at_fpr,
        # Explicit threshold-calibrated metrics for debugging and tables.
        "threshold_fixed": float(threshold),
        "threshold_at_fpr": float(thr_at_fpr),
        "accuracy_at_fpr": float(calibrated["accuracy"]),
        "precision_at_fpr": float(calibrated["precision"]),
        "recall_at_fpr_threshold": float(calibrated["recall"]),
        "fpr_at_fpr_threshold": float(calibrated["fpr"]),
        "fnr_at_fpr_threshold": float(calibrated["fnr"]),
        "f1_at_fpr_threshold": float(calibrated["f1"]),
    }
    return out


def _smooth_numpy_fpr_fnr(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    temperature: float,
) -> Tuple[float, float]:
    """Compute the paper's differentiable FPR/FNR proxies for risk state."""

    temp = max(float(temperature), 1e-6)
    smooth_pred = 1.0 / (1.0 + np.exp(-(probs - float(threshold)) / temp))
    neg = labels <= 0.5
    pos = labels > 0.5
    smooth_fpr = float(smooth_pred[neg].mean()) if np.any(neg) else 0.0
    smooth_fnr = float((1.0 - smooth_pred[pos]).mean()) if np.any(pos) else 0.0
    return smooth_fpr, smooth_fnr


def collect_probs_and_labels(
    net: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    net.eval()
    probs_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device).float()
            logits = net(images)
            probs = torch.sigmoid(logits)
            probs_list.append(probs.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())

    if not probs_list:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    return np.concatenate(probs_list).reshape(-1), np.concatenate(labels_list).reshape(-1)


def evaluate_binary_metrics(
    net: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    target_fpr: float,
    temperature: float | None = None,
) -> Dict[str, float]:
    probs, labels = collect_probs_and_labels(net, loader, device)
    if len(labels) == 0:
        return {}
    metrics = _hard_binary_metrics(probs, labels, threshold, target_fpr)
    metrics["anomaly_ratio"] = float(np.mean(labels))
    metrics["threshold"] = float(threshold)
    metrics["target_fpr"] = float(target_fpr)
    if temperature is not None:
        smooth_fpr, smooth_fnr = _smooth_numpy_fpr_fnr(
            probs, labels, threshold, temperature
        )
        metrics["smooth_fpr"] = smooth_fpr
        metrics["smooth_fnr"] = smooth_fnr
        metrics["smooth_fpr_violation"] = max(smooth_fpr - float(target_fpr), 0.0)
    return metrics


def _smooth_fpr_fnr(
    probs: torch.Tensor,
    labels: torch.Tensor,
    tau: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    neg_mask = labels <= 0.5
    pos_mask = labels > 0.5

    if neg_mask.any():
        smooth_fpr = torch.sigmoid((probs[neg_mask] - tau) / temperature).mean()
    else:
        smooth_fpr = torch.zeros((), device=probs.device)

    if pos_mask.any():
        smooth_fnr = torch.sigmoid((tau - probs[pos_mask]) / temperature).mean()
    else:
        smooth_fnr = torch.zeros((), device=probs.device)

    return smooth_fpr, smooth_fnr


def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    *,
    threshold: float = 0.5,
    lambda_fpr: float = 0.0,
    epsilon_fpr: float = 0.01,
    temperature: float = 0.05,
    c_fn: float = 5.0,
    c_fp: float = 1.0,
    global_anomaly_ratio: float = 0.10,
    beta_power: float = 0.5,
    beta_alpha: float = 1.0,
    beta_zeta: float = 1.0,
    beta_min: float = 1.0,
    beta_max: float = 20.0,
    beta_reference_ratio: float = 0.0,
    beta_gate_mode: str = "smooth",
    beta_round_decay: float = 0.0,
    mu_fnr: float = 0.2,
    eta_lambda: float = 1.0,
    lr_tau: float = 0.002,
    fpr_penalty: float = 2.0,
    server_round: int = 1,
    risk_weight_warmup_rounds: int = 3,
    risk_gamma_fnr: float = 0.5,
    risk_gamma_fpr: float = 0.5,
    risk_weight_min_factor: float = 0.5,
    risk_weight_max_factor: float = 2.0,
    loss_type: str = "bce",
    focal_gamma: float = 2.0,
    focal_alpha: float = -1.0,
    fedprox_mu: float = 0.0,
    global_params: Dict[str, torch.Tensor] | None = None,
    moon_mu: float = 0.0,
    moon_temperature: float = 0.5,
    global_model: nn.Module | None = None,
    previous_model: nn.Module | None = None,
    fedsimsup_mu: float = 0.0,
    fedsimsup_temperature: float = 0.5,
    supervisor_model: nn.Module | None = None,
) -> Tuple[float, List[float], Dict[str, float], float, float]:
    """Train local model with RC-FAD objective.

    Returns:
        final_loss, epoch_losses, risk_metrics, new_threshold, new_lambda_fpr
    """

    net.to(device)
    net.train()
    if global_model is not None:
        global_model.to(device)
        global_model.eval()
    if previous_model is not None:
        previous_model.to(device)
        previous_model.eval()
    if supervisor_model is not None:
        supervisor_model.to(device)
        supervisor_model.eval()
    if global_params is not None:
        global_params = {k: v.detach().to(device) for k, v in global_params.items()}

    # Pre-train risk state for the paper's dynamic minority enhancement.
    # The hard metrics remain useful diagnostics, but beta is driven by the
    # differentiable FPR/FNR proxies used in the theoretical method.
    pre_metrics = evaluate_binary_metrics(
        net, trainloader, device, threshold, epsilon_fpr, temperature
    )
    local_ratio = max(float(pre_metrics.get("anomaly_ratio", 0.0)), 1e-8)
    pre_hard_fnr = float(pre_metrics.get("fnr", 0.0))
    pre_hard_fpr = float(pre_metrics.get("fpr", 0.0))
    pre_smooth_fnr = float(pre_metrics.get("smooth_fnr", pre_hard_fnr))
    pre_smooth_fpr = float(pre_metrics.get("smooth_fpr", pre_hard_fpr))
    epsilon_safe = max(float(epsilon_fpr), 1e-8)
    global_ratio = max(float(global_anomaly_ratio), 1e-8)
    reference_ratio = max(global_ratio, float(beta_reference_ratio), 1e-8)
    scarcity = np.clip((reference_ratio - local_ratio) / reference_ratio, 0.0, 1.0)
    scarcity_gate = float(scarcity ** max(float(beta_power), 0.0))
    gate_mode = str(beta_gate_mode).lower()
    gate_fnr = pre_hard_fnr if gate_mode in {"hard", "hard_risk", "hard-risk"} else pre_smooth_fnr
    gate_fpr = pre_hard_fpr if gate_mode in {"hard", "hard_risk", "hard-risk"} else pre_smooth_fpr
    fnr_gate = 1.0 / (
        1.0 + np.exp(-float(beta_alpha) * (gate_fnr - epsilon_safe))
    )
    fpr_budget_margin = (epsilon_safe - gate_fpr) / epsilon_safe
    fpr_budget_gate = 1.0 / (
        1.0 + np.exp(-float(beta_zeta) * (fpr_budget_margin - 0.25))
    )
    if float(beta_round_decay) > 0.0:
        round_gate = max(
            0.0,
            1.0 - max(0, int(server_round) - 1) / max(float(beta_round_decay), 1.0),
        )
    else:
        round_gate = 1.0

    # Risk-gated hard-positive enhancement: increase minority pressure only
    # when positives are scarce, false negatives remain high, and FPR has room.
    beta_logit = scarcity_gate * fnr_gate * fpr_budget_gate * round_gate
    beta_k = float(beta_min + (beta_max - beta_min) * beta_logit)
    beta_k = float(np.clip(beta_k, beta_min, beta_max))

    tau = nn.Parameter(torch.tensor(float(threshold), dtype=torch.float32, device=device))
    optimizer = torch.optim.SGD(
        [
            {"params": net.parameters(), "lr": lr, "momentum": 0.9},
            {"params": [tau], "lr": lr_tau},
        ]
    )

    epoch_losses: List[float] = []

    for _ in range(epochs):
        running_loss = 0.0
        batches = 0

        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device).float()

            optimizer.zero_grad()
            logits = net(images)
            probs = torch.sigmoid(logits)

            # Stable weighted binary cross entropy:
            # y*softplus(-logit) = -y*log(sigmoid(logit))
            # (1-y)*softplus(logit) = -(1-y)*log(1-sigmoid(logit))
            with torch.no_grad():
                hard_positive_gate = torch.sigmoid((tau - probs) / max(float(temperature), 1e-6))
                positive_weight = float(c_fn) * (1.0 + (beta_k - 1.0) * hard_positive_gate)
            pos_loss = labels * F.softplus(-logits) * positive_weight
            neg_loss = (1.0 - labels) * F.softplus(logits) * c_fp
            sample_loss = pos_loss + neg_loss
            if str(loss_type).lower() == "focal":
                p_t = labels * probs + (1.0 - labels) * (1.0 - probs)
                focal = torch.pow(torch.clamp(1.0 - p_t, min=0.0), float(focal_gamma))
                if float(focal_alpha) >= 0.0:
                    alpha_t = labels * float(focal_alpha) + (1.0 - labels) * (1.0 - float(focal_alpha))
                    focal = focal * alpha_t
                sample_loss = sample_loss * focal
            risk_loss = sample_loss.mean()

            smooth_fpr, smooth_fnr = _smooth_fpr_fnr(probs, labels, tau, temperature)
            # Use a hinge-style low-FPR constraint. This is more stable than
            # the raw Lagrangian term lambda*(FPR-epsilon) in early rounds.
            fpr_violation = torch.relu(smooth_fpr - epsilon_fpr)
            objective = risk_loss + (lambda_fpr + fpr_penalty) * fpr_violation + mu_fnr * smooth_fnr
            if float(fedprox_mu) > 0.0 and global_params is not None:
                prox = torch.zeros((), device=device)
                for name, param in net.named_parameters():
                    if name in global_params:
                        prox = prox + torch.sum((param - global_params[name]) ** 2)
                objective = objective + 0.5 * float(fedprox_mu) * prox
            if (
                float(moon_mu) > 0.0
                and global_model is not None
                and previous_model is not None
            ):
                current_features = F.normalize(net.forward_features(images), dim=1)
                with torch.no_grad():
                    global_features = F.normalize(global_model.forward_features(images), dim=1)
                    previous_features = F.normalize(previous_model.forward_features(images), dim=1)
                pos_sim = torch.sum(current_features * global_features, dim=1)
                neg_sim = torch.sum(current_features * previous_features, dim=1)
                contrast_logits = torch.stack([pos_sim, neg_sim], dim=1) / max(float(moon_temperature), 1e-6)
                contrast_labels = torch.zeros(images.size(0), dtype=torch.long, device=device)
                objective = objective + float(moon_mu) * F.cross_entropy(contrast_logits, contrast_labels)
            if (
                float(fedsimsup_mu) > 0.0
                and supervisor_model is not None
                and images.size(0) > 1
            ):
                current_features = F.normalize(net.forward_features(images), dim=1)
                with torch.no_grad():
                    supervisor_features = F.normalize(supervisor_model.forward_features(images), dim=1)
                    teacher_sim = torch.matmul(supervisor_features, supervisor_features.T)
                    teacher_dist = F.softmax(
                        teacher_sim / max(float(fedsimsup_temperature), 1e-6),
                        dim=1,
                    )
                student_sim = torch.matmul(current_features, current_features.T)
                student_log_dist = F.log_softmax(
                    student_sim / max(float(fedsimsup_temperature), 1e-6),
                    dim=1,
                )
                simsup_loss = F.kl_div(student_log_dist, teacher_dist, reduction="batchmean")
                objective = objective + float(fedsimsup_mu) * simsup_loss

            objective.backward()
            optimizer.step()

            with torch.no_grad():
                tau.clamp_(0.0, 1.0)

            running_loss += float(objective.item())
            batches += 1

        epoch_losses.append(running_loss / max(1, batches))

    new_threshold = float(tau.detach().cpu().item())

    post_metrics = evaluate_binary_metrics(
        net, trainloader, device, new_threshold, epsilon_fpr, temperature
    )
    smooth_fpr = float(post_metrics.get("smooth_fpr", post_metrics.get("fpr", 0.0)))
    smooth_fnr = float(post_metrics.get("smooth_fnr", post_metrics.get("fnr", 0.0)))
    constraint_violation = max(smooth_fpr - float(epsilon_fpr), 0.0)
    new_lambda_fpr = max(
        0.0,
        float(lambda_fpr) + float(eta_lambda) * constraint_violation,
    )

    # Client-side fallback for risk-aware aggregation. The exact paper weight is
    # finalized on the server, where update reliability q_k can be computed.
    risk_focus = (1.0 + risk_gamma_fnr * smooth_fnr) * np.exp(
        -risk_gamma_fpr * constraint_violation
    )
    risk_focus = float(
        np.clip(risk_focus, risk_weight_min_factor, risk_weight_max_factor)
    )
    if int(server_round) <= int(risk_weight_warmup_rounds):
        risk_focus = 1.0
    reliability = 1.0 / (
        1.0 + float(np.var(epoch_losses)) if len(epoch_losses) > 1 else 1.0
    )
    risk_weight = len(trainloader.dataset) * risk_focus * reliability

    post_metrics.update({
        "train_loss": epoch_losses[-1],
        "beta": beta_k,
        "beta_logit": float(beta_logit),
        "beta_scarcity_gate": float(scarcity_gate),
        "beta_fnr_gate": float(fnr_gate),
        "beta_fpr_budget_gate": float(fpr_budget_gate),
        "beta_round_gate": float(round_gate),
        "lambda_fpr": new_lambda_fpr,
        "smooth_fpr": smooth_fpr,
        "smooth_fnr": smooth_fnr,
        "constraint_violation": constraint_violation,
        "loss_type": 0.0,
        "fedsimsup_mu": float(fedsimsup_mu),
        "risk_weight": float(risk_weight),
        "num-examples": float(len(trainloader.dataset)),
    })

    return epoch_losses[-1], epoch_losses, post_metrics, new_threshold, new_lambda_fpr


def test(
    net: nn.Module,
    testloader: DataLoader,
    device: torch.device,
    *,
    threshold: float = 0.5,
    target_fpr: float = 0.01,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate binary anomaly detection model."""

    net.to(device)
    net.eval()

    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device).float()
            logits = net(images)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            total_loss += float(loss.item())
            batches += 1

    metrics = evaluate_binary_metrics(net, testloader, device, threshold, target_fpr)
    metrics["eval_loss"] = total_loss / max(1, batches)
    # Compatibility alias used by old logging/checkpoint code
    metrics["eval_acc"] = metrics.get("accuracy", 0.0)
    return metrics["eval_loss"], metrics
