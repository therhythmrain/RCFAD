"""Compatibility facade for the RC-FAD Flower apps.

The original quickstart kept model, data loading, metrics, and training in one
large ``task.py`` file.  The implementation now lives in focused modules while
this file preserves the old import surface used by ``client_app.py`` and
``server_app.py``.
"""

from RCFAD.constants import TABULAR_CACHE_VERSION, TABULAR_INPUT_DIM
from RCFAD.data import (
    BinaryAnomalyDataset,
    NormalScoreShiftDataset,
    TabularAnomalyDataset,
    _is_tabular_dataset,
    _load_tabular_arrays,
    load_centralized_dataset,
    load_data,
)
from RCFAD.metrics import (
    _basic_hard_metrics,
    _compute_auc,
    _compute_auprc,
    _hard_binary_metrics,
    _next_score_after_max,
    _recall_at_fpr,
    _safe_div,
    _smooth_numpy_fpr_fnr,
    _threshold_at_fpr,
    collect_probs_and_labels,
    evaluate_binary_metrics,
)
from RCFAD.model import Net
from RCFAD.training import _smooth_fpr_fnr, test, train
from RCFAD.utils import TrainProcessMetadata, set_seed

__all__ = [
    "TABULAR_CACHE_VERSION",
    "TABULAR_INPUT_DIM",
    "TrainProcessMetadata",
    "Net",
    "BinaryAnomalyDataset",
    "NormalScoreShiftDataset",
    "TabularAnomalyDataset",
    "load_data",
    "load_centralized_dataset",
    "_is_tabular_dataset",
    "_load_tabular_arrays",
    "collect_probs_and_labels",
    "evaluate_binary_metrics",
    "train",
    "test",
    "set_seed",
]
