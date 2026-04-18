"""Train a word-level ASL classifier from 90-frame, 2-hand landmark clips.

Reads data/landmarks_words.csv produced by capture.py's TAB/word mode.
Each row is: label, then WORD_CLIP_FRAMES * 2 * 21 * 3 = 11340 raw landmark
floats. Two hand slots per frame, leftmost wrist x first; missing hands are
zero-padded.

Legacy 1-hand rows (length WORD_CLIP_FRAMES * 21 * 3 = 5670 floats) from the
pre-2-hand capture format are auto-migrated by stuffing the data into slot 1
(right) and zeroing slot 0. This means old samples are interpreted as
"one-handed sign performed by the right hand" — which matches what most
solo-hand demos look like on a mirrored webcam. Mirror augmentation handles
the symmetric case.

Feature design (per sample):
  - Pick N_KEYFRAMES evenly-spaced frames from the clip.
  - For each keyframe, emit:
      * 2 * 90-dim shape features (one per hand slot, zero block for missing).
      * 2 * 2-dim wrist offset (dx, dy) per slot, vs slot's own frame-0 wrist.
        A missing slot's offset is zeros.
  => feature length = N_KEYFRAMES * 2 * (90 + 2)

Augmentations:
  - Mirror (x -> 1-x on every landmark of every hand). Slot order swaps too,
    so the leftmost-first invariant is preserved.
  - Time jitter: shift the keyframe sampling window by +/-3 frames.

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
from app.services.features import (  # noqa: E402
    HANDS_PER_FRAME,
    SHAPE_FEATURE_LEN,
    WORD_FEATURE_VERSION,
    build_two_hand_shape_features,
    pad_hands_to_two,
    sort_hands_by_x,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_OUT = ROOT / "models" / "asl_words_classifier.joblib"
DATA_DIR = ROOT.parent / "data"
CSV_WORDS = DATA_DIR / "landmarks_words.csv"

WORD_CLIP_FRAMES = 90       # must match capture.py WORD_FRAMES
N_KEYFRAMES = 18            # every 5 frames
TRAJ_DIMS_PER_HAND = 2      # (dx, dy) wrist offset per slot
PER_KEYFRAME_DIMS = HANDS_PER_FRAME * (SHAPE_FEATURE_LEN + TRAJ_DIMS_PER_HAND)

TIME_JITTERS = (-3, 0, 3)   # keyframe-window offsets for augmentation

FLOATS_PER_HAND_FRAME = 21 * 3
TWO_HAND_ROW_FLOATS = WORD_CLIP_FRAMES * HANDS_PER_FRAME * FLOATS_PER_HAND_FRAME
ONE_HAND_ROW_FLOATS = WORD_CLIP_FRAMES * FLOATS_PER_HAND_FRAME


def read_words_csv(path: Path) -> Iterable[Tuple[str, np.ndarray]]:
    """Yield (label, (WORD_CLIP_FRAMES, HANDS_PER_FRAME, 21, 3) array)."""
    if not path.exists():
        raise SystemExit(f"No word data at {path}. Run capture.py with TAB/word mode first.")
    legacy_count = 0
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            label = row[0]
            floats = np.array(row[1:], dtype=np.float32)
            if floats.size == TWO_HAND_ROW_FLOATS:
                clip = floats.reshape(WORD_CLIP_FRAMES, HANDS_PER_FRAME, 21, 3)
            elif floats.size == ONE_HAND_ROW_FLOATS:
                # Legacy 1-hand row: stuff data into slot 1 (right), zero slot 0.
                one = floats.reshape(WORD_CLIP_FRAMES, 21, 3)
                clip = np.zeros((WORD_CLIP_FRAMES, HANDS_PER_FRAME, 21, 3), dtype=np.float32)
                clip[:, 1] = one
                legacy_count += 1
            else:
                print(
                    f"  skipping {label}: got {floats.size} floats, "
                    f"expected {TWO_HAND_ROW_FLOATS} (2-hand) or {ONE_HAND_ROW_FLOATS} (legacy 1-hand)"
                )
                continue
            yield label, clip
    if legacy_count:
        print(f"  migrated {legacy_count} legacy 1-hand rows into slot 1")


def keyframe_indices(offset: int = 0) -> np.ndarray:
    base = np.linspace(0, WORD_CLIP_FRAMES - 1, N_KEYFRAMES).round().astype(int)
    return np.clip(base + offset, 0, WORD_CLIP_FRAMES - 1)


def _frame_to_canonical_hands(frame: np.ndarray) -> List[np.ndarray]:
    """Take a (HANDS_PER_FRAME, 21, 3) frame and return slot-sorted, padded
    hand arrays. Slots that were all-zero (missing) stay all-zero after sort."""
    real = [h for h in frame if not np.all(h == 0)]
    sorted_real = sort_hands_by_x(real)
    return pad_hands_to_two(sorted_real)


def clip_to_features(clip: np.ndarray, offset: int = 0) -> np.ndarray:
    """Build the per-clip feature vector. `clip` is (frames, 2, 21, 3)."""
    idxs = keyframe_indices(offset)

    # Reference wrist xy per slot at the first keyframe — used so each slot's
    # trajectory is measured relative to that slot's own starting position.
    first_hands = _frame_to_canonical_hands(clip[idxs[0]])
    slot_ref_xy = [h[0, :2].copy() for h in first_hands]
    slot_present_at_start = [not bool(np.all(h == 0)) for h in first_hands]

    per_frame: List[np.ndarray] = []
    for i in idxs:
        hands = _frame_to_canonical_hands(clip[i])
        shape = build_two_hand_shape_features(hands)
        traj_parts: List[float] = []
        for slot, h in enumerate(hands):
            if np.all(h == 0) or not slot_present_at_start[slot]:
                traj_parts.extend([0.0, 0.0])
            else:
                traj_parts.append(float(h[0, 0] - slot_ref_xy[slot][0]))
                traj_parts.append(float(h[0, 1] - slot_ref_xy[slot][1]))
        per_frame.append(np.concatenate([shape, np.asarray(traj_parts, dtype=np.float32)]))
    return np.concatenate(per_frame).astype(np.float32)


def mirror_clip(clip: np.ndarray) -> np.ndarray:
    """Flip x for every landmark of every hand. Then re-sort hand slots so the
    leftmost-first invariant still holds (mirrored slot 0 may now be rightmost).

    Important: zero-padded (missing) hand slots must stay zero after the flip.
    Naive `x = 1 - x` turns (0,0,0) landmarks into (1,0,0) garbage, which
    destroys the model's ability to tell "slot empty" from "slot has a hand",
    erasing the main signal that distinguishes 1-hand vs 2-hand signs.
    """
    out = clip.copy()
    # Remember which (frame, slot) entries were empty so we can restore them.
    zero_mask = np.all(clip == 0, axis=(2, 3))  # shape (frames, hands_per_frame)
    out[..., 0] = 1.0 - out[..., 0]
    out[zero_mask] = 0.0
    # Re-sort per frame by wrist x of non-zero hands.
    for f in range(out.shape[0]):
        frame = out[f]
        real = [h for h in frame if not np.all(h == 0)]
        sorted_real = sort_hands_by_x(real)
        padded = pad_hands_to_two(sorted_real)
        for s, h in enumerate(padded):
            out[f, s] = h
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
            "word_feature_version": WORD_FEATURE_VERSION,
            "clip_frames": WORD_CLIP_FRAMES,
            "n_keyframes": N_KEYFRAMES,
            "per_keyframe_dims": PER_KEYFRAME_DIMS,
            "hands_per_frame": HANDS_PER_FRAME,
        },
        MODEL_OUT,
    )
    print(f"Saved word model to {MODEL_OUT} (word_feature_version={WORD_FEATURE_VERSION})")


if __name__ == "__main__":
    main()
