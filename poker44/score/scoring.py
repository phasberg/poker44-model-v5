"""Reward and scoring utilities for Poker44 poker bot detection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix


def reward(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, dict]:
    """
    Compute a reward that strongly protects humans while still rewarding bot recall.
    """
    preds = np.round(y_pred).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    negative_count = max(tn + fp, 1)
    positive_count = max(tp + fn, 1)

    fpr = fp / negative_count
    bot_recall = tp / positive_count

    if y_pred.size and np.any(y_true == 1):
        ap_score = average_precision_score(y_true, y_pred)
    else:
        ap_score = 0.0

    # Hard human-safety penalty: a miner that harms humans should not win.
    human_safety_penalty = max(0.0, 1.0 - fpr) ** 2
    if fpr >= 0.10:
        human_safety_penalty = 0.0

    base_score = 0.65 * ap_score + 0.35 * bot_recall
    rew = base_score * human_safety_penalty

    res = {
        "fpr": fpr,
        "bot_recall": bot_recall,
        "ap_score": ap_score,
        "human_safety_penalty": human_safety_penalty,
        "base_score": base_score,
        "reward": rew,
    }
    return rew, res
