from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np
import joblib

from app.services.features import FEATURE_VERSION, build_features

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
STATIC_MODEL_PATH = MODELS_DIR / "asl_classifier.joblib"
MOTION_MODEL_PATH = MODELS_DIR / "asl_motion_classifier.joblib"

# Number of recent probability vectors to average for the static classifier.
# Filters per-frame noise without making transitions sluggish.
SMOOTH_WINDOW = 8

# Motion routing. The rolling landmark buffer holds the last CLIP_FRAMES raw
# 21x3 arrays; the motion classifier reads the whole buffer when it fires.
CLIP_FRAMES = 70                  # must match train_motion.py / capture.py
MOTION_WINDOW = 15                # sliding-window size for motion detection
MOTION_DISPLACEMENT_THRESHOLD = 0.008  # stationary hand ~0.0005, J/Z peak 0.03-0.06
SETTLE_TAIL_FRAMES = 6            # frames of recent calm to trust the motion is done
SETTLE_DISPLACEMENT = 0.005       # displacement threshold considered "calm"

_static_bundle = None
_motion_bundle = None
_probs_buffer: Deque[np.ndarray] = deque(maxlen=SMOOTH_WINDOW)
_landmark_buffer: Deque[np.ndarray] = deque(maxlen=CLIP_FRAMES)
# Cached motion result — once fired, we keep returning it until the motion
# leaves the buffer, rather than re-running inference every frame.
_last_motion_result: Tuple[str, float] | None = None


def _load_static():
    global _static_bundle
    if _static_bundle is None:
        if not STATIC_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Classifier model not found at {STATIC_MODEL_PATH}. "
                "Run `python training/train.py` to create one."
            )
        _static_bundle = joblib.load(STATIC_MODEL_PATH)
        saved_ver = _static_bundle.get("feature_version")
        if saved_ver != FEATURE_VERSION:
            raise RuntimeError(
                f"Static model at {STATIC_MODEL_PATH} was trained with feature_version="
                f"{saved_ver}, but this server expects {FEATURE_VERSION}. "
                "Retrain with `python training/train.py ...`."
            )
    return _static_bundle


def _load_motion():
    """Motion model is optional — if absent, we silently stay static-only."""
    global _motion_bundle
    if _motion_bundle is None and MOTION_MODEL_PATH.exists():
        bundle = joblib.load(MOTION_MODEL_PATH)
        if bundle.get("feature_version") != FEATURE_VERSION:
            print(
                f"warn: motion model feature_version {bundle.get('feature_version')} "
                f"!= expected {FEATURE_VERSION}; ignoring motion model"
            )
            return None
        if bundle.get("clip_frames") != CLIP_FRAMES:
            print(
                f"warn: motion model clip_frames {bundle.get('clip_frames')} "
                f"!= server clip_frames {CLIP_FRAMES}; ignoring motion model"
            )
            return None
        _motion_bundle = bundle
    return _motion_bundle


def _mean_landmark_displacement(frames: List[np.ndarray]) -> float:
    """Mean magnitude of per-landmark (x, y) change between consecutive frames,
    averaged over all 21 landmarks."""
    if len(frames) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(frames[:-1], frames[1:]):
        diffs = b[:, :2] - a[:, :2]
        total += float(np.linalg.norm(diffs, axis=1).mean())
    return total / (len(frames) - 1)


def _motion_features(buffer: List[np.ndarray]) -> np.ndarray:
    """Rebuild the same flat feature vector that train_motion.py emits."""
    n_keyframes = _motion_bundle["n_keyframes"]
    idxs = np.linspace(0, CLIP_FRAMES - 1, n_keyframes).round().astype(int)
    frame0_wrist_xy = buffer[idxs[0]][0, :2]

    parts: List[np.ndarray] = []
    for i in idxs:
        frame = buffer[i]
        shape = build_features(frame)
        wrist_dx = frame[0, 0] - frame0_wrist_xy[0]
        wrist_dy = frame[0, 1] - frame0_wrist_xy[1]
        parts.append(np.concatenate([shape, [wrist_dx, wrist_dy]]).astype(np.float32))
    return np.concatenate(parts).reshape(1, -1)


def _classify_static(points_array: np.ndarray) -> Tuple[str, float]:
    bundle = _load_static()
    model = bundle["model"]
    labels = bundle["labels"]
    features = build_features(points_array).reshape(1, -1)
    probs = model.predict_proba(features)[0]

    _probs_buffer.append(probs)
    smoothed = np.mean(_probs_buffer, axis=0)
    idx = int(np.argmax(smoothed))
    return labels[idx], float(smoothed[idx])


def _classify_motion() -> Tuple[str, float]:
    bundle = _motion_bundle
    model = bundle["model"]
    labels = bundle["labels"]
    features = _motion_features(list(_landmark_buffer))
    probs = model.predict_proba(features)[0]
    idx = int(np.argmax(probs))
    return labels[idx], float(probs[idx])


def _buffer_has_motion(buf: List[np.ndarray]) -> bool:
    """True if any MOTION_WINDOW-frame window inside the buffer exceeds the
    motion threshold. Scans every 5 frames for speed."""
    for start in range(0, len(buf) - MOTION_WINDOW + 1, 5):
        if _mean_landmark_displacement(buf[start:start + MOTION_WINDOW]) > MOTION_DISPLACEMENT_THRESHOLD:
            return True
    return False


def classify_landmarks(points: List[Tuple[float, float, float]]) -> Tuple[str, float]:
    """Always runs the static classifier so the frontend sees a stream of
    letters. If the landmark buffer contains a completed motion (buffer full,
    recent frames calm, earlier frames moving), the motion model's output
    overrides the static prediction."""
    global _last_motion_result

    arr = np.asarray(points, dtype=np.float32).reshape(21, 3)
    _landmark_buffer.append(arr)

    motion_bundle = _load_motion()
    buf = list(_landmark_buffer)

    if motion_bundle is not None and len(buf) == CLIP_FRAMES:
        tail_calm = _mean_landmark_displacement(buf[-SETTLE_TAIL_FRAMES:]) < SETTLE_DISPLACEMENT
        had_motion = _buffer_has_motion(buf)

        if tail_calm and had_motion:
            if _last_motion_result is None:
                _last_motion_result = _classify_motion()
            return _last_motion_result
        if not had_motion:
            _last_motion_result = None

    return _classify_static(arr)


def reset_smoothing() -> None:
    """Frontend calls this when the hand leaves the frame so a stale sign
    doesn't bleed into the next one."""
    global _last_motion_result
    _probs_buffer.clear()
    _landmark_buffer.clear()
    _last_motion_result = None
