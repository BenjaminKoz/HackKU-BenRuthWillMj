from pathlib import Path
from typing import List, Tuple

import numpy as np
import joblib

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "asl_classifier.joblib"

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Classifier model not found at {MODEL_PATH}. "
                "Run `python training/train.py` to create one."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def normalize_landmarks(points: List[Tuple[float, float, float]]) -> np.ndarray:
    """Translate to wrist origin and scale by hand size for translation/scale invariance."""
    arr = np.array(points, dtype=np.float32)  # (21, 3)
    wrist = arr[0].copy()
    arr -= wrist
    middle_mcp = arr[9]
    scale = float(np.linalg.norm(middle_mcp)) or 1.0
    arr /= scale
    return arr.flatten()  # (63,)


def classify_landmarks(points: List[Tuple[float, float, float]]) -> Tuple[str, float]:
    bundle = _load()
    model = bundle["model"]
    labels = bundle["labels"]
    features = normalize_landmarks(points).reshape(1, -1)
    probs = model.predict_proba(features)[0]
    idx = int(np.argmax(probs))
    return labels[idx], float(probs[idx])
