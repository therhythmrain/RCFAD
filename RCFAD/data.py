"""Dataset loading, preprocessing, caching, and client partitioning."""

from __future__ import annotations

import os
import re
import ssl
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor

from RCFAD.constants import TABULAR_CACHE_VERSION, TABULAR_INPUT_DIM


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
        "Use cifar10, mnist, fmnist, creditcard, baf, ai4i, secom, swat, hai, "
        "smd, paysim, mammography, annthyroid, shuttle, or tep."
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
        "swat": "swat",
        "securewatertreatment": "swat",
        "hai": "hai",
        "haicon": "hai",
        "smd": "smd",
        "servermachinedataset": "smd",
        "paysim": "paysim",
        "paysim1": "paysim",
        "mobilemoneyfraud": "paysim",
        "mammography": "mammography",
        "oddsmammography": "mammography",
        "annthyroid": "annthyroid",
        "oddsannthyroid": "annthyroid",
        "thyroid": "annthyroid",
        "shuttle": "shuttle",
        "oddsshuttle": "shuttle",
        "tep": "tep",
        "tennesseeeastman": "tep",
        "tennesseeeastmanprocess": "tep",
    }
    return aliases.get(name, name)


def _is_tabular_dataset(dataset_name: str) -> bool:
    return _normalize_dataset_key(dataset_name) in {
        "creditcard",
        "baf",
        "ai4i",
        "secom",
        "swat",
        "hai",
        "smd",
        "paysim",
        "mammography",
        "annthyroid",
        "shuttle",
        "tep",
    }


def _urlretrieve_with_ssl_fallback(url: str, path: str) -> None:
    try:
        urllib.request.urlretrieve(url, path)
        return
    except Exception as exc:
        if "CERTIFICATE_VERIFY_FAILED" not in str(exc):
            raise
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=context) as response, open(path, "wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _download_and_extract_zip(url: str, zip_path: str, extract_dir: str) -> None:
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    print(f"Downloading {url} to {zip_path} ...")
    _urlretrieve_with_ssl_fallback(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)


def _download_file(url: str, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) <= 0:
        os.remove(path)
    if not os.path.exists(path):
        print(f"Downloading {url} to {path} ...")
        _urlretrieve_with_ssl_fallback(url, path)
    return path


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
            if lower.endswith((".csv", ".csv.gz")) and all(key in lower for key in keys):
                return os.path.join(dirpath, filename)
    return None


def _find_first_file(root: str, suffixes: Iterable[str], keywords: Iterable[str] = ()) -> str | None:
    suffix_list = tuple(suffix.lower() for suffix in suffixes)
    keys = [key.lower() for key in keywords]
    if not os.path.exists(root):
        return None
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            lower = filename.lower()
            if suffix_list and not lower.endswith(suffix_list):
                continue
            if all(key in lower for key in keys):
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


def _ensure_adbench_npz(data_root: str, key: str) -> str:
    root = os.path.abspath(data_root)
    urls = {
        "mammography": "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/23_mammography.npz",
        "annthyroid": "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/2_annthyroid.npz",
        "shuttle": "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/32_shuttle.npz",
    }
    filenames = {
        "mammography": "23_mammography.npz",
        "annthyroid": "2_annthyroid.npz",
        "shuttle": "32_shuttle.npz",
    }
    candidates = [
        os.path.join(root, key, filenames[key]),
        os.path.join(root, key, f"{key}.npz"),
        os.path.join(root, filenames[key]),
        os.path.join(root, f"{key}.npz"),
    ]
    found = _first_existing_path(candidates)
    if found is not None:
        return found
    found = _find_first_file(os.path.join(root, key), (".npz",), (key,))
    if found is not None:
        return found
    return _download_file(urls[key], os.path.join(root, key, filenames[key]))


def _load_adbench_npz_arrays(data_root: str, key: str) -> Tuple[np.ndarray, np.ndarray]:
    npz_path = _ensure_adbench_npz(data_root, key)
    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.files)
    feature_key = "X" if "X" in keys else "x" if "x" in keys else "data" if "data" in keys else ""
    label_key = "y" if "y" in keys else "Y" if "Y" in keys else "label" if "label" in keys else "labels" if "labels" in keys else ""
    if not feature_key or not label_key:
        raise ValueError(f"Unsupported ADBench npz layout in {npz_path}: keys={data.files}")
    features = np.asarray(data[feature_key], dtype=np.float32)
    labels = np.asarray(data[label_key]).reshape(-1)
    labels = (labels > 0).astype(np.float32)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError(
            f"Invalid ADBench arrays in {npz_path}: features={features.shape}, labels={labels.shape}"
        )
    return features, labels


def _ensure_paysim_csv(data_root: str) -> str:
    root = os.path.abspath(data_root)
    candidates = [
        os.path.join(root, "paysim", "PS_20174392719_1491204439457_log.csv"),
        os.path.join(root, "paysim", "paysim.csv"),
        os.path.join(root, "paysim.csv"),
    ]
    found = _first_existing_path(candidates)
    if found is not None:
        return found
    found = _find_first_csv(os.path.join(root, "paysim"), [])
    if found is not None:
        return found
    raise FileNotFoundError(
        "PaySim dataset not found. Download PaySim from Kaggle and place the "
        "CSV under data/paysim/PS_20174392719_1491204439457_log.csv or data/paysim.csv."
    )


def _ensure_timeseries_csv(data_root: str, key: str) -> str:
    root = os.path.abspath(data_root)
    candidates = [
        os.path.join(root, key, f"{key}.csv"),
        os.path.join(root, key, f"{key.upper()}.csv"),
        os.path.join(root, f"{key}.csv"),
        os.path.join(root, f"{key.upper()}.csv"),
    ]
    if key == "swat":
        candidates.extend([
            os.path.join(root, "swat", "merged.csv"),
            os.path.join(root, "swat", "attack.csv"),
            os.path.join(root, "swat", "SWaT_Dataset_Attack_v0.csv"),
            os.path.join(root, "swat", "SWaT_Dataset_Normal_v1.csv"),
            os.path.join(root, "SWaT_Dataset_Attack_v0.csv"),
        ])
    if key == "hai":
        candidates.extend([
            os.path.join(root, "hai", "train.csv"),
            os.path.join(root, "hai", "test.csv"),
            os.path.join(root, "hai", "HAI.csv"),
        ])
    found = _first_existing_path(candidates)
    if found is not None:
        return found
    found = _find_first_csv(os.path.join(root, key), [])
    if found is not None:
        return found
    raise FileNotFoundError(
        f"{key.upper()} dataset not found. Place a CSV under data/{key}/ with "
        "a label column such as label, attack, anomaly, is_anomaly, or attack_label."
    )


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


def _find_label_column(df: pd.DataFrame) -> str:
    candidates = [
        "label",
        "attack",
        "anomaly",
        "is_anomaly",
        "isattack",
        "is_attack",
        "attack_label",
        "normal/attack",
        "normal_attack",
        "class",
        "target",
    ]
    lower_to_col = {str(col).strip().lower(): col for col in df.columns}
    for name in candidates:
        if name in lower_to_col:
            return lower_to_col[name]
    raise ValueError(f"No label column found. Expected one of {candidates}. Columns: {list(df.columns)[:30]}")


def _labels_from_series(series: pd.Series) -> np.ndarray:
    normal_values = {"0", "normal", "benign", "false", "no", "none"}
    return series.map(lambda value: 0.0 if str(value).strip().lower() in normal_values else 1.0).to_numpy(dtype=np.float32)


def _load_timeseries_csv_arrays(data_root: str, key: str) -> Tuple[np.ndarray, np.ndarray]:
    if key == "hai":
        hai_root = os.path.join(os.path.abspath(data_root), "hai")
        paths = []
        for dirpath, _, filenames in os.walk(hai_root):
            for filename in sorted(filenames):
                lower = filename.lower()
                if lower.endswith((".csv", ".csv.gz")):
                    paths.append(os.path.join(dirpath, filename))
        if not paths:
            paths = [_ensure_timeseries_csv(data_root, key)]
        frames = [pd.read_csv(path, sep=None, engine="python") for path in sorted(paths)]
        df = pd.concat(frames, ignore_index=True)
    else:
        csv_path = _ensure_timeseries_csv(data_root, key)
        df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [str(col).strip() for col in df.columns]
    label_col = _find_label_column(df)
    labels = _labels_from_series(df[label_col])
    drops = [label_col]
    for col in ("timestamp", "time", "date", "datetime"):
        if col in {c.lower() for c in df.columns}:
            drops.append(col)
    features, _ = _dataframe_to_features(df, label_col=label_col, positive_values=(1,), drop_cols=drops)
    return features, labels


def _ensure_tep_files(data_root: str) -> Tuple[List[str], List[str]]:
    root = os.path.abspath(data_root)
    tep_root = os.path.join(root, "tep")
    normal_candidates = [
        os.path.join(tep_root, "mode1_normal_50.xlsx"),
        os.path.join(tep_root, "normal.xlsx"),
        os.path.join(tep_root, "normal.csv"),
    ]
    fault_candidates = [
        os.path.join(tep_root, "mode1_1_1.xlsx"),
        os.path.join(tep_root, "mode1_2_1.xlsx"),
    ]

    normal_path = _first_existing_path(normal_candidates)
    if normal_path is None:
        normal_path = _download_file(
            "https://github.com/mv-per/tennessee-eastman-dataset/raw/main/simulations/mode_1/mode1_normal_50.xlsx",
            normal_candidates[0],
        )

    existing_faults = [path for path in fault_candidates if os.path.exists(path)]
    if not existing_faults:
        fault_urls = [
            "https://github.com/mv-per/tennessee-eastman-dataset/raw/main/simulations/mode_1/faults/mode1_1_1.xlsx",
            "https://github.com/mv-per/tennessee-eastman-dataset/raw/main/simulations/mode_1/faults/mode1_2_1.xlsx",
        ]
        existing_faults = [
            _download_file(url, path)
            for url, path in zip(fault_urls, fault_candidates)
        ]

    extra_faults = []
    for dirpath, _, filenames in os.walk(tep_root):
        for filename in filenames:
            lower = filename.lower()
            path = os.path.join(dirpath, filename)
            if path == normal_path or path in existing_faults:
                continue
            if lower.endswith((".xlsx", ".xls", ".csv")) and ("fault" in lower or "mode1_" in lower):
                extra_faults.append(path)
    fault_paths = sorted(set(existing_faults + extra_faults))
    if not fault_paths:
        raise FileNotFoundError(
            "TEP fault files not found. Place fault xlsx/csv files under data/tep/ "
            "or allow automatic download from mv-per/tennessee-eastman-dataset."
        )
    return [normal_path], fault_paths


def _read_tep_table(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".csv"):
        df = pd.read_csv(path, sep=None, engine="python")
    else:
        try:
            df = pd.read_excel(path)
        except ImportError:
            df = _read_xlsx_table_stdlib(path)
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _xlsx_col_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref.upper())
    if not match:
        return 0
    idx = 0
    for char in match.group(1):
        idx = idx * 26 + (ord(char) - ord("A") + 1)
    return idx - 1


def _read_xlsx_table_stdlib(path: str) -> pd.DataFrame:
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall("main:si", ns):
                texts = [node.text or "" for node in item.findall(".//main:t", ns)]
                shared.append("".join(texts))

        sheet_names = [name for name in zf.namelist() if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")]
        if not sheet_names:
            raise ValueError(f"No worksheets found in {path}")
        root = ET.fromstring(zf.read(sorted(sheet_names)[0]))

    rows: List[List[Any]] = []
    for row in root.findall(".//main:sheetData/main:row", ns):
        values: List[Any] = []
        for cell in row.findall("main:c", ns):
            col_idx = _xlsx_col_index(cell.attrib.get("r", "A1"))
            while len(values) <= col_idx:
                values.append("")
            cell_type = cell.attrib.get("t", "")
            value_node = cell.find("main:v", ns)
            inline_node = cell.find("main:is/main:t", ns)
            raw = ""
            if value_node is not None and value_node.text is not None:
                raw = value_node.text
            elif inline_node is not None and inline_node.text is not None:
                raw = inline_node.text
            if cell_type == "s" and raw:
                raw = shared[int(raw)]
            values[col_idx] = raw
        rows.append(values)

    if not rows:
        raise ValueError(f"Worksheet is empty: {path}")
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    header = [str(value).strip() or f"col_{idx}" for idx, value in enumerate(padded[0])]
    df = pd.DataFrame(padded[1:], columns=header)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        non_empty = df[col].astype(str).str.strip().ne("")
        if int(converted.notna().sum()) == int(non_empty.sum()):
            df[col] = converted
    return df


def _load_tep_arrays(data_root: str) -> Tuple[np.ndarray, np.ndarray]:
    root = os.path.abspath(data_root)
    labelled_csv = _find_first_csv(os.path.join(root, "tep"), ["tep"])
    if labelled_csv is not None:
        df = pd.read_csv(labelled_csv, sep=None, engine="python")
        df.columns = [str(col).strip() for col in df.columns]
        try:
            label_col = _find_label_column(df)
        except ValueError:
            label_col = ""
        if label_col:
            labels = _labels_from_series(df[label_col])
            features, _ = _dataframe_to_features(df, label_col=label_col, positive_values=(1,))
            return features, labels

    normal_paths, fault_paths = _ensure_tep_files(data_root)
    frames: List[pd.DataFrame] = []
    for path in normal_paths:
        df = _read_tep_table(path)
        df["__rcfad_label__"] = 0
        frames.append(df)
    for path in fault_paths:
        df = _read_tep_table(path)
        df["__rcfad_label__"] = 1
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    features, labels = _dataframe_to_features(
        combined,
        label_col="__rcfad_label__",
        positive_values=(1,),
        drop_cols=("time", "timestamp", "datetime", "date", "sample", "simulationRun"),
    )
    return features, labels


def _read_smd_matrix(path: str) -> np.ndarray:
    lower = path.lower()
    if lower.endswith(".npy"):
        arr = np.load(path)
    else:
        try:
            arr = pd.read_csv(path, header=None).to_numpy()
        except Exception:
            arr = pd.read_csv(path, sep=r"\s+", header=None).to_numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _ensure_smd_files(data_root: str) -> Tuple[str, str, str | None]:
    root = os.path.abspath(data_root)
    smd_root = os.path.join(root, "smd")
    train_candidates = [
        os.path.join(smd_root, "train", "machine-1-1.txt"),
        os.path.join(smd_root, "train", "machine-1-1.csv"),
        os.path.join(smd_root, "train.csv"),
        os.path.join(root, "SMD", "train", "machine-1-1.txt"),
    ]
    test_candidates = [
        os.path.join(smd_root, "test", "machine-1-1.txt"),
        os.path.join(smd_root, "test", "machine-1-1.csv"),
        os.path.join(smd_root, "test.csv"),
        os.path.join(root, "SMD", "test", "machine-1-1.txt"),
    ]
    label_candidates = [
        os.path.join(smd_root, "test_label", "machine-1-1.txt"),
        os.path.join(smd_root, "test_label", "machine-1-1.csv"),
        os.path.join(smd_root, "test_label.csv"),
        os.path.join(root, "SMD", "test_label", "machine-1-1.txt"),
    ]
    train_path = _first_existing_path(train_candidates) or _find_first_file(os.path.join(smd_root, "train"), (".txt", ".csv", ".npy"))
    test_path = _first_existing_path(test_candidates) or _find_first_file(os.path.join(smd_root, "test"), (".txt", ".csv", ".npy"))
    label_path = _first_existing_path(label_candidates) or _find_first_file(os.path.join(smd_root, "test_label"), (".txt", ".csv", ".npy"))
    if train_path is not None and test_path is not None:
        return train_path, test_path, label_path
    csv_path = _find_first_csv(smd_root, [])
    if csv_path is not None:
        return csv_path, csv_path, None
    raise FileNotFoundError(
        "SMD dataset not found. Place files under data/smd/train, data/smd/test, "
        "and data/smd/test_label, or provide a labelled CSV under data/smd/."
    )


def _window_timeseries(features: np.ndarray, labels: np.ndarray, window_size: int = 10, stride: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    if len(features) != len(labels):
        min_len = min(len(features), len(labels))
        features = features[:min_len]
        labels = labels[:min_len]
    if len(features) < window_size:
        return features, labels
    xs: List[np.ndarray] = []
    ys: List[float] = []
    for start in range(0, len(features) - window_size + 1, stride):
        end = start + window_size
        xs.append(features[start:end].reshape(-1))
        ys.append(float(labels[start:end].max()))
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32)


def _load_smd_arrays(data_root: str) -> Tuple[np.ndarray, np.ndarray]:
    train_path, test_path, label_path = _ensure_smd_files(data_root)
    if label_path is None and train_path == test_path and train_path.lower().endswith(".csv"):
        df = pd.read_csv(train_path)
        label_col = _find_label_column(df)
        labels = _labels_from_series(df[label_col])
        features, _ = _dataframe_to_features(df, label_col=label_col, positive_values=(1,))
        return _window_timeseries(features, labels)
    train_x = _read_smd_matrix(train_path)
    test_x = _read_smd_matrix(test_path)
    train_y = np.zeros(len(train_x), dtype=np.float32)
    if label_path is None:
        test_y = np.zeros(len(test_x), dtype=np.float32)
    else:
        test_y = _read_smd_matrix(label_path).reshape(-1).astype(np.float32)
        test_y = (test_y > 0).astype(np.float32)
    x = np.concatenate([train_x, test_x], axis=0)
    y = np.concatenate([train_y, test_y], axis=0)
    return _window_timeseries(x, y)


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


def _max_rows_for_dataset(key: str) -> int:
    if key != "paysim":
        return 0
    value = os.environ.get("RCFAD_PAYSIM_MAX_ROWS", "100000").strip()
    try:
        return max(0, int(value))
    except ValueError:
        return 100000


def _stratified_row_cap(
    features: np.ndarray,
    labels: np.ndarray,
    max_rows: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    max_rows = int(max_rows)
    if max_rows <= 0 or len(labels) <= max_rows:
        return features, labels
    rng = np.random.default_rng(int(seed))
    labels = labels.astype(np.float32, copy=False)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    pos_target = max(1, int(round(max_rows * len(pos_idx) / len(labels)))) if len(pos_idx) else 0
    neg_target = max_rows - pos_target
    pos_keep = rng.choice(pos_idx, size=min(pos_target, len(pos_idx)), replace=False) if len(pos_idx) else np.array([], dtype=int)
    neg_keep = rng.choice(neg_idx, size=min(neg_target, len(neg_idx)), replace=False) if len(neg_idx) else np.array([], dtype=int)
    keep = rng.permutation(np.concatenate([pos_keep, neg_keep]))
    return features[keep], labels[keep]


def _tabular_cache_dir(data_root: str, key: str, seed: int) -> str:
    suffix = ""
    max_rows = _max_rows_for_dataset(key)
    if max_rows > 0:
        suffix = f"_cap{max_rows}"
    return os.path.join(
        os.path.abspath(data_root),
        ".rcfad_cache",
        f"{key}_seed{int(seed)}_v{TABULAR_CACHE_VERSION}{suffix}",
    )


def _tabular_cache_paths(data_root: str, key: str, seed: int) -> dict[str, str]:
    cache_dir = _tabular_cache_dir(data_root, key, seed)
    return {
        "dir": cache_dir,
        "lock": f"{cache_dir}.lock",
        "ready": os.path.join(cache_dir, "READY"),
        "x_train": os.path.join(cache_dir, "x_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "x_test": os.path.join(cache_dir, "x_test.npy"),
        "y_test": os.path.join(cache_dir, "y_test.npy"),
    }


def _load_tabular_cache(data_root: str, key: str, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    paths = _tabular_cache_paths(data_root, key, seed)
    required = ("ready", "x_train", "y_train", "x_test", "y_test")
    if not all(os.path.exists(paths[name]) for name in required):
        return None
    return (
        np.load(paths["x_train"], mmap_mode="r"),
        np.load(paths["y_train"], mmap_mode="r"),
        np.load(paths["x_test"], mmap_mode="r"),
        np.load(paths["y_test"], mmap_mode="r"),
    )


def _write_tabular_cache(
    data_root: str,
    key: str,
    seed: int,
    arrays: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    paths = _tabular_cache_paths(data_root, key, seed)
    os.makedirs(paths["dir"], exist_ok=True)
    for name, array in zip(("x_train", "y_train", "x_test", "y_test"), arrays):
        tmp_path = f"{paths[name]}.tmp.{os.getpid()}"
        with open(tmp_path, "wb") as f:
            np.save(f, np.asarray(array))
        os.replace(tmp_path, paths[name])
    with open(paths["ready"], "w", encoding="utf-8") as f:
        f.write("ok\n")


def _acquire_tabular_cache_lock(data_root: str, key: str, seed: int, timeout: float = 3600.0) -> str:
    paths = _tabular_cache_paths(data_root, key, seed)
    os.makedirs(os.path.dirname(paths["lock"]), exist_ok=True)
    start = time.time()
    while True:
        try:
            fd = os.open(paths["lock"], os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"pid={os.getpid()}\n")
            return paths["lock"]
        except FileExistsError:
            if _load_tabular_cache(data_root, key, seed) is not None:
                return ""
            if time.time() - start > timeout:
                try:
                    os.remove(paths["lock"])
                except OSError:
                    pass
                start = time.time()
            time.sleep(2.0)


def _build_tabular_arrays(
    key: str,
    data_root: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    elif key in {"mammography", "annthyroid", "shuttle"}:
        features, labels = _load_adbench_npz_arrays(data_root, key)
    elif key in {"swat", "hai"}:
        features, labels = _load_timeseries_csv_arrays(data_root, key)
        features, labels = _window_timeseries(features, labels)
    elif key == "smd":
        features, labels = _load_smd_arrays(data_root)
    elif key == "tep":
        features, labels = _load_tep_arrays(data_root)
    elif key == "paysim":
        csv_path = _ensure_paysim_csv(data_root)
        df = pd.read_csv(
            csv_path,
            usecols=[
                "step",
                "type",
                "amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "oldbalanceDest",
                "newbalanceDest",
                "isFraud",
            ],
        )
        features, labels = _dataframe_to_features(
            df,
            label_col="isFraud",
            positive_values=(1, True, "true"),
        )
        features, labels = _stratified_row_cap(
            features,
            labels,
            _max_rows_for_dataset(key),
            int(seed),
        )
    else:
        raise ValueError(f"Unsupported tabular dataset {key!r}.")

    return _split_standardize_tabular(features, labels, int(seed))


@lru_cache(maxsize=16)
def _load_tabular_arrays(
    dataset_name: str,
    data_root: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load, encode, split, and normalize supported tabular anomaly datasets.

    Large tabular/time-series datasets are cached as ``.npy`` arrays and loaded
    with mmap in each Flower/Ray client actor. This preserves the 10-client
    experimental setting without forcing every actor to parse and hold a full
    CSV/DataFrame copy in memory.
    """

    key = _normalize_dataset_key(dataset_name)
    cached = _load_tabular_cache(data_root, key, int(seed))
    if cached is not None:
        return cached

    lock_path = _acquire_tabular_cache_lock(data_root, key, int(seed))
    try:
        cached = _load_tabular_cache(data_root, key, int(seed))
        if cached is not None:
            return cached
        arrays = _build_tabular_arrays(key, data_root, int(seed))
        _write_tabular_cache(data_root, key, int(seed), arrays)
        return _load_tabular_cache(data_root, key, int(seed)) or arrays
    finally:
        if lock_path:
            try:
                os.remove(lock_path)
            except OSError:
                pass


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
