from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np
import joblib

from app.services.features import FEATURE_VERSION, build_features

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "asl_classifier.joblib"

# Number of recent probability vectors to average. At ~30fps this is ~0.3s of
# context — long enough to filter per-frame noise, short enough that transitions
# between signs still feel responsive.
SMOOTH_WINDOW = 8

_bundle = None
_probs_buffer: Deque[np.ndarray] = deque(maxlen=SMOOTH_WINDOW)


def _load():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Classifier model not found at {MODEL_PATH}. "
                "Run `python training/train.py` to create one."
            )
        _bundle = joblib.load(MODEL_PATH)
        saved_ver = _bundle.get("feature_version")
        if saved_ver != FEATURE_VERSION:
            raise RuntimeError(
                f"Model at {MODEL_PATH} was trained with feature_version="
                f"{saved_ver}, but this server expects {FEATURE_VERSION}. "
                "Retrain with `python training/train.py ...`."
            )
    return _bundle


def classify_landmarks(points: List[Tuple[float, float, float]]) -> Tuple[str, float]:
    bundle = _load()
    model = bundle["model"]
    labels = bundle["labels"]
    features = build_features(points).reshape(1, -1)
    probs = model.predict_proba(features)[0]

    _probs_buffer.append(probs)
    smoothed = np.mean(_probs_buffer, axis=0)
    idx = int(np.argmax(smoothed))
    return labels[idx], float(smoothed[idx])


def reset_smoothing() -> None:
    """Clear the probability buffer. The frontend should call this when the hand
    leaves the frame so a stale sign doesn't bleed into the next one."""
    _probs_buffer.clear()
