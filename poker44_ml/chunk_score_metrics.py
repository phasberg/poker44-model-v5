"""Chunk-level probability diagnostics (training / sequence / trees)."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)

from poker44.score.scoring import reward


def enrich_chunk_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    raw_scores: Sequence[float] | None = None,
) -> Dict[str, float]:
    """Same family of metrics as tree training in ``train_model_v2`` / ``train_model``."""
    safe = [max(1e-6, min(1.0 - 1e-6, float(v))) for v in scores]
    metrics: Dict[str, float] = {}
    label_list = [int(v) for v in labels]
    if len(set(label_list)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(label_list, safe))
        metrics["pr_auc"] = float(average_precision_score(label_list, safe))
        metrics["mcc_at_0_5"] = float(
            matthews_corrcoef(label_list, [v >= 0.5 for v in safe])
        )
    metrics["log_loss"] = float(log_loss(label_list, safe, labels=[0, 1]))
    metrics["prob_min"] = float(min(safe)) if safe else 0.0
    metrics["prob_max"] = float(max(safe)) if safe else 0.0
    metrics["prob_mean"] = float(sum(safe) / max(len(safe), 1))

    preds = [v >= 0.5 for v in safe]
    tp = sum(1 for label, pred in zip(label_list, preds) if label == 1 and pred)
    fp = sum(1 for label, pred in zip(label_list, preds) if label == 0 and pred)
    tn = sum(1 for label, pred in zip(label_list, preds) if label == 0 and not pred)
    fn = sum(1 for label, pred in zip(label_list, preds) if label == 1 and not pred)
    positives = max(sum(1 for label in label_list if label == 1), 1)
    negatives = max(sum(1 for label in label_list if label == 0), 1)
    metrics["recall_at_0_5"] = float(tp / positives)
    metrics["precision_at_0_5"] = float(tp / max(tp + fp, 1))
    metrics["fpr_at_0_5"] = float(fp / negatives)

    val_reward, details = reward(
        np.asarray(safe, dtype=float),
        np.asarray(label_list, dtype=int),
    )
    metrics["validator_reward"] = float(val_reward)
    metrics["validator_fpr"] = float(details.get("fpr", 1.0))
    metrics["validator_bot_recall"] = float(details.get("bot_recall", 0.0))
    metrics["validator_ap_score"] = float(details.get("ap_score", 0.0))

    humans = [s for label, s in zip(label_list, safe) if label == 0]
    bots = [s for label, s in zip(label_list, safe) if label == 1]
    metrics["human_prob_max"] = float(max(humans)) if humans else 0.0
    metrics["bot_prob_min"] = float(min(bots)) if bots else 1.0
    metrics["score_gap_at_0_5"] = metrics["bot_prob_min"] - metrics["human_prob_max"]

    if raw_scores is not None:
        raw = [max(0.0, min(1.0, float(v))) for v in raw_scores]
        raw_humans = [s for label, s in zip(label_list, raw) if label == 0]
        raw_bots = [s for label, s in zip(label_list, raw) if label == 1]
        metrics["raw_prob_min"] = float(min(raw)) if raw else 0.0
        metrics["raw_prob_max"] = float(max(raw)) if raw else 0.0
        metrics["raw_prob_mean"] = float(sum(raw) / max(len(raw), 1))
        metrics["raw_human_prob_max"] = float(max(raw_humans)) if raw_humans else 0.0
        metrics["raw_bot_prob_min"] = float(min(raw_bots)) if raw_bots else 1.0
        metrics["raw_score_gap_at_0_5"] = (
            metrics["raw_bot_prob_min"] - metrics["raw_human_prob_max"]
        )
    return metrics


def human_bot_prob_bounds(
    labels: Sequence[int],
    scores: Sequence[float],
) -> Dict[str, float]:
    """Max human / min bot score for a single score band (pre- or post-remap)."""
    label_list = [int(v) for v in labels]
    safe = [max(0.0, min(1.0, float(v))) for v in scores]
    humans = [s for label, s in zip(label_list, safe) if label == 0]
    bots = [s for label, s in zip(label_list, safe) if label == 1]
    human_max = float(max(humans)) if humans else 0.0
    bot_min = float(min(bots)) if bots else 1.0
    return {
        "human_prob_max": human_max,
        "bot_prob_min": bot_min,
        "score_gap_at_0_5": bot_min - human_max,
    }


def format_chunk_metrics_line(metrics: Dict[str, float]) -> str:
    parts = [
        f"pr_auc={metrics.get('pr_auc', 0.0):.4f}",
        f"reward={metrics.get('validator_reward', 0.0):.4f}",
        f"fpr={metrics.get('validator_fpr', 0.0):.4f}",
        f"bot_recall={metrics.get('validator_bot_recall', 0.0):.4f}",
        f"prob_min={metrics.get('prob_min', 0.0):.4f}",
        f"prob_max={metrics.get('prob_max', 0.0):.4f}",
        f"human_prob_max={metrics.get('human_prob_max', 0.0):.4f}",
        f"bot_prob_min={metrics.get('bot_prob_min', 0.0):.4f}",
    ]
    if "raw_prob_min" in metrics:
        parts.extend(
            [
                f"raw_min={metrics['raw_prob_min']:.4f}",
                f"raw_max={metrics['raw_prob_max']:.4f}",
                f"raw_human_prob_max={metrics.get('raw_human_prob_max', 0.0):.4f}",
                f"raw_bot_prob_min={metrics.get('raw_bot_prob_min', 0.0):.4f}",
            ]
        )
    return " ".join(parts)


def print_chunk_score_diagnostics(
    title: str,
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    raw_scores: Sequence[float] | None = None,
    indent: str = "  ",
) -> Dict[str, float]:
    metrics = enrich_chunk_metrics(labels, scores, raw_scores=raw_scores)
    print(f"{indent}{title}: {format_chunk_metrics_line(metrics)}")
    return metrics
