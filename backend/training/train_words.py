"""Train a word-level ASL classifier from 90-frame landmark clips.

Reads data/landmarks_words.csv produced by capture.py's TAB/word mode.
Each row is: label, then WORD_CLIP_FRAMES * 63 raw landmark floats.

Feature design (per sample):
  - Pick N_KEYFRAMES evenly-spaced frames from the clip.
  - For each keyframe, emit:
      * The 90-dim normalized hand-shape feature vector from build_features.
        Captures hand pose at that instant.
      * A 2-dim wrist offset (dx, dy) from frame 0's wrist in image space.
        Per-frame normalization strips translation, so we add this back
        separately — it's what encodes trajectory across the sign.
  => feature length = N_KEYFRAMES * (90 + 2)

This is the same recipe as train_motion.py. The knobs that differ for words:
WORD_CLIP_FRAMES is longer (~3s), and N_KEYFRAMES is higher so multi-phase
signs (e.g. HELLO's salute sweep, THANKS's chin-out motion) are sampled
densely enough to separate.

Augmentations:
  - Mirror (x -> 1-x on every frame) for hand-invariance.
  - Time jitter: shift the keyframe sampling window by +/-3 frames to tolerate
    variation in when the sign starts within the clip.

Usage:
    python training/train_words.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.features import FEATURE_VERSION, build_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_OUT = ROOT / "models" / "asl_words_classifier.joblib"
DATA_DIR = ROOT.parent / "data"
CSV_WORDS = DATA_DIR / "landmarks_words.csv"

WORD_CLIP_FRAMES = 90       # must match capture.py WORD_FRAMES
N_KEYFRAMES = 18            # every 5 frames
SHAPE_DIMS = 90             # length of build_features() output
TRAJ_DIMS = 2               # (dx, dy) wrist offset
PER_KEYFRAME_DIMS = SHAPE_DIMS + TRAJ_DIMS

TIME_JITTERS = (-3, 0, 3)   # keyframe-window offsets for augmentation


def read_words_csv(path: Path) -> Iterable[Tuple[str, np.ndarray]]:
    """Yield (label, (WORD_CLIP_FRAMES, 21, 3) array)."""
    if not path.exists():
        raise SystemExit(f"No word data at {path}. Run capture.py with TAB/word mode first.")
    expected_floats = WORD_CLIP_FRAMES * 63
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            label = row[0]
            floats = np.array(row[1:], dtype=np.float32)
            if floats.size != expected_floats:
                print(f"  skipping {label}: got {floats.size} floats, expected {expected_floats}")
                continue
            yield label, floats.reshape(WORD_CLIP_FRAMES, 21, 3)


def keyframe_indices(offset: int = 0) -> np.ndarray:
    base = np.linspace(0, WORD_CLIP_FRAMES - 1, N_KEYFRAMES).round().astype(int)
    return np.clip(base + offset, 0, WORD_CLIP_FRAMES - 1)


def clip_to_features(clip: np.ndarray, offset: int = 0) -> np.ndarray:
    idxs = keyframe_indices(offset)
    frame0_wrist_xy = clip[idxs[0], 0, :2]

    per_frame: List[np.ndarray] = []
    for i in idxs:
        frame = clip[i]
        shape = build_features(frame)
        wrist_dx = frame[0, 0] - frame0_wrist_xy[0]
        wrist_dy = frame[0, 1] - frame0_wrist_xy[1]
        per_frame.append(np.concatenate([shape, [wrist_dx, wrist_dy]]).astype(np.float32))
    return np.concatenate(per_frame)


def mirror_clip(clip: np.ndarray) -> np.ndarray:
    out = clip.copy()
    out[..., 0] = 1.0 - out[..., 0]
    return out


def load_dataset(mirror_augment: bool = True, time_jitter: bool = True):
    X: List[np.ndarray] = []
    y: List[str] = []
    raw_counts: dict[str, int] = {}

    offsets = TIME_JITTERS if time_jitter else (0,)

    for label, clip in read_words_csv(CSV_WORDS):
        raw_counts[label] = raw_counts.get(label, 0) + 1
        variants = [clip]
        if mirror_augment:
            variants.append(mirror_clip(clip))
        for v in variants:
            for off in offsets:
                X.append(clip_to_features(v, offset=off))
                y.append(label)

    if not X:
        raise SystemExit("No word samples found.")
    factor = (2 if mirror_augment else 1) * len(offsets)
    print(f"Loaded raw samples per label: {raw_counts}")
    print(f"After {factor}x augmentation (mirror={mirror_augment}, time_jitter={time_jitter}): "
          f"{len(X)} total rows, {len(X[0])} features each")
    return np.asarray(X, dtype=np.float32), y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--no-mirror-augment", action="store_true")
    parser.add_argument("--no-time-jitter", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    X, y = load_dataset(
        mirror_augment=not args.no_mirror_augment,
        time_jitter=not args.no_time_jitter,
    )
    labels = sorted(set(y))
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    y_idx = np.array([label_to_idx[lbl] for lbl in y])

    do_split = min(np.bincount(y_idx)) >= 5
    if do_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_idx, test_size=args.test_size, random_state=42, stratify=y_idx
        )
    else:
        print("Too few samples per class for a test split — training on everything.")
        X_train, y_train = X, y_idx
        X_test, y_test = X, y_idx

    clf = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "labels": labels,
            "feature_version": FEATURE_VERSION,
            "clip_frames": WORD_CLIP_FRAMES,
            "n_keyframes": N_KEYFRAMES,
            "per_keyframe_dims": PER_KEYFRAME_DIMS,
        },
        MODEL_OUT,
    )
    print(f"Saved word model to {MODEL_OUT} (feature_version={FEATURE_VERSION})")


if __name__ == "__main__":
    main()
