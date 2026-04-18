from pathlib import Path
from typing import List, Tuple

import numpy as np
import joblib

from app.services.features import FEATURE_VERSION, build_features

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
    idx = int(np.argmax(probs))
    return labels[idx], float(probs[idx])
