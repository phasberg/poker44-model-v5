"""Stacked ensemble for Poker44 with isotonic post-calibration.

This class is the single inference object stored under ``artifact["models"]``
so the existing :class:`poker44_ml.inference.Poker44Model` loader picks it up
unchanged. It exposes a ``predict_proba`` method that internally:

1. Selects an optional feature subset (top-K important features).
2. Runs each feature-based base learner (LightGBM, XGBoost, CatBoost,
   ExtraTrees, RandomForest).
3. Optionally runs chunk-based base learners (e.g. the Set Transformer
   sequence model) on the raw chunk payloads through
   :meth:`predict_chunk_scores`.
4. Stacks all base probabilities through a logistic-regression meta-learner
   trained on out-of-fold base predictions.
5. Applies isotonic calibration so output is monotone for ranking-based AP.

A global ``score_shift`` (typically a small negative number) is conformally
fitted on a held-out set to keep chunk-level FPR strictly below the validator
cliff at 10 percent (default target 4 percent).
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Sequence

import numpy as np

# Cosmetic LightGBM<->sklearn 1.7 feature-name warning. Filtered here so the
# warning does not appear when the stacked ensemble is unpickled and used
# inside the miner inference path.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)


class StackedEnsemble:
    """Picklable stacked ensemble compatible with the Poker44Model loader.

    Two kinds of base learners are supported:

    * ``base_models`` consume aligned feature rows (numpy matrix), the same
      input that the canonical Poker44Model loader currently provides.
    * ``chunk_models`` consume the **raw chunk payloads** (lists of hand
      dicts) and must expose ``predict_proba(chunks) -> Nx2`` (or a
      ``predict_chunk_scores(chunks)`` method). These run only via
      :meth:`predict_chunk_scores`.

    For backward compatibility the canonical ``predict_proba(rows)`` path is
    still supported when no chunk models are present.
    """

    def __init__(
        self,
        base_models: Sequence[Any],
        meta_model: Any,
        calibrator: Optional[Any] = None,
        feature_indices: Optional[Sequence[int]] = None,
        score_shift: float = 0.0,
        chunk_models: Optional[Sequence[Any]] = None,
    ) -> None:
        self.base_models: List[Any] = list(base_models)
        self.chunk_models: List[Any] = list(chunk_models or [])
        self.meta_model = meta_model
        self.calibrator = calibrator
        self.feature_indices: Optional[np.ndarray] = (
            np.asarray(list(feature_indices), dtype=np.int64)
            if feature_indices is not None
            else None
        )
        self.score_shift = float(score_shift)

    def _select_features(self, x: np.ndarray) -> np.ndarray:
        if self.feature_indices is None:
            return x
        return x[:, self.feature_indices]

    def _base_probs(self, x: np.ndarray) -> np.ndarray:
        cols: List[np.ndarray] = []
        for model in self.base_models:
            if hasattr(model, "predict_proba"):
                proba = np.asarray(model.predict_proba(x))
                col = proba[:, 1] if proba.ndim == 2 else proba
            elif hasattr(model, "decision_function"):
                raw = np.asarray(model.decision_function(x), dtype=float)
                col = 1.0 / (1.0 + np.exp(-np.clip(raw, -40.0, 40.0)))
            else:
                col = np.asarray(model.predict(x), dtype=float)
            cols.append(np.clip(np.asarray(col, dtype=float), 0.0, 1.0))
        return np.stack(cols, axis=1)

    def _chunk_probs(self, chunks: Sequence[Any]) -> np.ndarray:
        if not self.chunk_models:
            return np.zeros((len(chunks), 0), dtype=float)
        cols: List[np.ndarray] = []
        for model in self.chunk_models:
            if hasattr(model, "predict_proba"):
                proba = np.asarray(model.predict_proba(chunks))
                col = proba[:, 1] if proba.ndim == 2 else proba
            elif hasattr(model, "predict_chunk_scores"):
                col = np.asarray(model.predict_chunk_scores(chunks), dtype=float)
            else:
                raise RuntimeError(
                    f"Chunk model {type(model).__name__} exposes neither "
                    "predict_proba nor predict_chunk_scores."
                )
            cols.append(np.clip(np.asarray(col, dtype=float), 0.0, 1.0))
        return np.stack(cols, axis=1)

    def base_score_matrix(self, x: np.ndarray) -> np.ndarray:
        """Expose feature-based base-model probabilities (diagnostics)."""
        x_arr = np.asarray(x, dtype=np.float64)
        x_sel = self._select_features(x_arr)
        return self._base_probs(x_sel)

    def _meta_to_output(self, z: np.ndarray) -> np.ndarray:
        meta_proba = np.asarray(self.meta_model.predict_proba(z))
        p1 = meta_proba[:, 1] if meta_proba.ndim == 2 else meta_proba
        if self.calibrator is not None:
            if hasattr(self.calibrator, "transform"):
                p1 = np.asarray(self.calibrator.transform(p1), dtype=float)
            elif hasattr(self.calibrator, "predict"):
                p1 = np.asarray(self.calibrator.predict(p1), dtype=float)
        if self.score_shift:
            p1 = self._logit_shift(p1, self.score_shift)
        return np.clip(p1, 0.0, 1.0)

    def predict_proba(self, x: Any) -> np.ndarray:
        if self.chunk_models:
            raise RuntimeError(
                "StackedEnsemble has chunk-based learners; use "
                "predict_chunk_scores(chunks) instead of predict_proba(rows)."
            )
        x_arr = np.asarray(x, dtype=np.float64)
        x_sel = self._select_features(x_arr)
        z = self._base_probs(x_sel)
        p1 = self._meta_to_output(z)
        return np.stack([1.0 - p1, p1], axis=1)

    def predict_chunk_scores(
        self,
        chunks: Sequence[Any],
        feature_rows: Any,
    ) -> List[float]:
        """Score raw chunks using pre-computed feature rows.

        ``feature_rows`` must be an iterable of feature lists already aligned
        with the artifact's ``feature_names`` ordering (the canonical
        Poker44Model loader supplies these via ``_aligned_rows``).
        """
        x_arr = np.asarray(feature_rows, dtype=np.float64)
        x_sel = self._select_features(x_arr)
        if self.base_models:
            feature_probs = self._base_probs(x_sel)
        else:
            feature_probs = np.zeros((len(chunks), 0), dtype=float)
        chunk_probs = self._chunk_probs(list(chunks))
        if feature_probs.size == 0 and chunk_probs.size == 0:
            raise RuntimeError("No base or chunk models are available for scoring.")
        if chunk_probs.size == 0:
            stacked = feature_probs
        elif feature_probs.size == 0:
            stacked = chunk_probs
        else:
            stacked = np.concatenate([feature_probs, chunk_probs], axis=1)
        p1 = self._meta_to_output(stacked)
        return [float(value) for value in p1]

    def predict(self, x: Any) -> np.ndarray:
        proba = self.predict_proba(x)[:, 1]
        return (proba >= 0.5).astype(int)

    @staticmethod
    def _logit_shift(values: np.ndarray, shift: float) -> np.ndarray:
        clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped)) + float(shift)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
