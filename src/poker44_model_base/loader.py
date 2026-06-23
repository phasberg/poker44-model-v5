from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import joblib
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("joblib is required to load model artifacts") from exc


class ModelArtifactNotFound(FileNotFoundError):
    """Raised when no model artifact is available locally."""


def resolve_artifact_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the model artifact path."""

    candidates: list[Path] = []
    if path:
        candidates.append(Path(path).expanduser())
    env_path = os.getenv("POKER44_MODEL_ARTIFACT") or os.getenv("POKER44_MODEL_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(Path(__file__).resolve().parents[2] / "artifacts" / "current.joblib")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise ModelArtifactNotFound(f"No model artifact found. Checked: {checked}")


def load_model(path: str | os.PathLike[str] | None = None) -> Any:
    """Load a model artifact."""

    artifact_path = resolve_artifact_path(path)
    return joblib.load(artifact_path)
