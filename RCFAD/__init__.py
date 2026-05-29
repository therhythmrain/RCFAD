"""RC-FAD package.

Set simulation and deterministic CUDA defaults before submodules import torch.
"""

from __future__ import annotations

import os

os.environ.setdefault("RAY_STARTUP_TIMEOUT_S", "300")
os.environ.setdefault("RAY_DISABLE_BOOTSTRAP", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
