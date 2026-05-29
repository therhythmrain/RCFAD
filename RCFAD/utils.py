"""Small shared utilities for reproducible RC-FAD runs."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch

# Configure deterministic/CUDA-related settings before most project imports.
os.environ.setdefault("RAY_STARTUP_TIMEOUT_S", "300")
os.environ.setdefault("RAY_DISABLE_BOOTSTRAP", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set Python/NumPy/PyTorch seeds for reproducible FL simulations."""
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
