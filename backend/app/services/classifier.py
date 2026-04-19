from collections import deque
from pathlib import Path
from typing import Deque, List, Sequence, Tuple

import numpy as np
import joblib

from app.services.features import (
    FEATURE_VERSION,
    HANDS_PER_FRAME,
    WORD_FEATURE_VERSION,
    build_features,
    build_two_hand_shape_features,
    pad_hands_to_two,
    sort_hands_by_x,
)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
STATIC_MODEL_PATH = MODELS_DIR / "asl_classifier.joblib"
WORDS_MODEL_PATH = MODELS_DIR / "asl_words_classifier.joblib"

# Number of recent probability vectors to average for the static classifier.
# Filters per-frame noise without making transitions sluggish.
SMOOTH_WINDOW = 8

WORD_CLIP_FRAMES = 90             # must match train_words.py / capture.py

_static_bundle = None
_words_bundle = None
_probs_buffer: Deque[np.ndarray] = deque(maxlen=SMOOTH_WINDOW)


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


def _load_words():
    """Word model is optional — word classification is a no-op without it."""
    global _words_bundle
    if _words_bundle is None and WORDS_MODEL_PATH.exists():
        bundle = joblib.load(WORDS_MODEL_PATH)
        saved_ver = bundle.get("word_feature_version")
        if saved_ver != WORD_FEATURE_VERSION:
            print(
                f"warn: word model word_feature_version {saved_ver} "
                f"!= expected {WORD_FEATURE_VERSION}; ignoring word model. "
                "Retrain with `python training/train_words.py`."
            )
            return None
        if bundle.get("clip_frames") != WORD_CLIP_FRAMES:
            print(
                f"warn: word model clip_frames {bundle.get('clip_frames')} "
                f"!= server clip_frames {WORD_CLIP_FRAMES}; ignoring word model"
            )
            return None
        if bundle.get("hands_per_frame") != HANDS_PER_FRAME:
            print(
                f"warn: word model hands_per_frame {bundle.get('hands_per_frame')} "
                f"!= expected {HANDS_PER_FRAME}; ignoring word model"
            )
            return None
        _words_bundle = bundle
    return _words_bundle


def _word_clip_features(
    hands_per_frame: List[List[np.ndarray]],
    bundle: dict,
) -> np.ndarray:
    """Rebuild the flat feature vector that train_words.py emits.

    `hands_per_frame[i]` is a list of 0-2 real hand arrays (21,3) detected in
    frame i. We sort leftmost-wrist first and pad missing slots with zeros.
    """
    n_keyframes = bundle["n_keyframes"]
    idxs = np.linspace(0, WORD_CLIP_FRAMES - 1, n_keyframes).round().astype(int)

    def canonical_at(i: int) -> List[np.ndarray]:
        sorted_real = sort_hands_by_x(hands_per_frame[i])
        return pad_hands_to_two(sorted_real)

    first_hands = canonical_at(int(idxs[0]))
    slot_ref_xy = [h[0, :2].copy() for h in first_hands]
    slot_present_at_start = [not bool(np.all(h == 0)) for h in first_hands]

    parts: List[np.ndarray] = []
    for i in idxs:
        hands = canonical_at(int(i))
        shape = build_two_hand_shape_features(hands)
        traj: List[float] = []
        for slot, h in enumerate(hands):
            if np.all(h == 0) or not slot_present_at_start[slot]:
                traj.extend([0.0, 0.0])
            else:
                traj.append(float(h[0, 0] - slot_ref_xy[slot][0]))
                traj.append(float(h[0, 1] - slot_ref_xy[slot][1]))
        parts.append(np.concatenate([shape, np.asarray(traj, dtype=np.float32)]))
    return np.concatenate(parts).reshape(1, -1).astype(np.float32)


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


def classify_landmarks(
    points: List[Tuple[float, float, float]],
    mode: str = "letters",
) -> Tuple[str, float]:
    """Per-frame letter classification.

    Word classification is NOT handled here — words go through
    classify_word_clip() because we want training-inference parity (the caller
    supplies an explicit 90-frame clip the way capture.py recorded them).
    """
    arr = np.asarray(points, dtype=np.float32).reshape(21, 3)
    return _classify_static(arr)


def classify_word_clip(
    frames: Sequence[Sequence[Sequence[Tuple[float, float, float]]]],
) -> Tuple[str, float]:
    """Classify a deliberate, caller-supplied word clip.

    `frames` must have WORD_CLIP_FRAMES entries. Each entry is a list of 0, 1,
    or 2 detected hands; each hand is 21 (x, y, z) landmarks. Hands are sorted
    leftmost-wrist first and missing slots are zero-padded to feed a fixed-size
    feature vector. This mirrors how train_words.py encodes each frame.
    """
    bundle = _load_words()
    if bundle is None:
        raise FileNotFoundError(
            f"Word model not found or incompatible at {WORDS_MODEL_PATH}. "
            "Run `python training/train_words.py` first."
        )
    if len(frames) != WORD_CLIP_FRAMES:
        raise ValueError(
            f"Expected {WORD_CLIP_FRAMES} frames, got {len(frames)}."
        )

    hands_per_frame: List[List[np.ndarray]] = []
    for i, frame in enumerate(frames):
        if len(frame) > HANDS_PER_FRAME:
            raise ValueError(
                f"Frame {i}: got {len(frame)} hands, max is {HANDS_PER_FRAME}."
            )
        frame_hands: List[np.ndarray] = []
        for hand in frame:
            arr = np.asarray(hand, dtype=np.float32).reshape(21, 3)
            frame_hands.append(arr)
        hands_per_frame.append(frame_hands)

    features = _word_clip_features(hands_per_frame, bundle)

    model = bundle["model"]
    labels = bundle["labels"]
    probs = model.predict_proba(features)[0]
    idx = int(np.argmax(probs))
    return labels[idx], float(probs[idx])


def reset_smoothing() -> None:
    """Frontend calls this when the hand leaves the frame so a stale sign
    doesn't bleed into the next one."""
    _probs_buffer.clear()


def get_word_labels() -> List[str]:
    """Return the sorted word-model labels, or [] if the model isn't loaded."""
    bundle = _load_words()
    if bundle is None:
        return []
    return list(bundle.get("labels", []))
