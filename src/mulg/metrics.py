"""Explicit MOSI/MOSEI scoring, including both binary neutral conventions."""

import numpy as np


def class7(values):
    """Clip to [-3, 3], NumPy round-to-even, then encode as 0..6."""
    return np.rint(np.clip(np.asarray(values), -3, 3)).astype(np.int64) + 3


def evaluate(y, scores, classes=None):
    y, scores = np.asarray(y).reshape(-1), np.asarray(scores).reshape(-1)
    if y.size == 0 or y.shape != scores.shape:
        raise ValueError("Expected equally sized, nonempty targets and predictions")
    if not np.isfinite(y).all() or not np.isfinite(scores).all():
        raise ValueError("Nonfinite targets/predictions")
    predicted = class7(scores) if classes is None else np.asarray(classes).reshape(-1)
    if predicted.shape != y.shape or not np.isin(predicted, np.arange(7)).all():
        raise ValueError("Expected one class in 0..6 per target")
    nonzero = y != 0
    acc2 = float(100 * np.mean((scores[nonzero] > 0) == (y[nonzero] > 0))) if nonzero.any() else None
    corr = None
    if y.size > 1 and y.std() > 0 and scores.std() > 0:
        corr = float(np.corrcoef(y, scores)[0, 1])
    return {
        "Acc2": acc2,
        "Acc2_with_zero": float(100 * np.mean((scores >= 0) == (y >= 0))),
        "Acc7": float(100 * np.mean(predicted == class7(y))),
        "MAE": float(np.mean(np.abs(y - scores))),
        "Corr": corr,
        "n_test": int(y.size),
        "n_binary_nonzero": int(nonzero.sum()),
    }
