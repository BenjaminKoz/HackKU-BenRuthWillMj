"""Train an ASL alphabet classifier from MediaPipe hand landmarks.

Data sources (can be combined — the script loads whichever exist):
  1) data/landmarks.csv           — from training/capture.py (your own webcam samples)
  2) data/landmarks_images.csv    — cached landmarks extracted from --images folder
  3) --images PATH                — Kaggle ASL Alphabet-style folder of images;
                                    landmarks are extracted on first run and cached
                                    to data/landmarks_images.csv so re-runs are fast.

Usage:
    python training/train.py
    python training/train.py --images ../data/asl_alphabet_train
    python training/train.py --images ../data/asl_alphabet_train --samples-per-class 200
    python training/train.py --skip-labels nothing,space,del
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Ensure `app.services.features` is importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.features import FEATURE_VERSION, build_features, mirror_landmarks  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_OUT = ROOT / "models" / "asl_classifier.joblib"
DATA_DIR = ROOT.parent / "data"
CSV_CAPTURE = DATA_DIR / "landmarks.csv"
CSV_IMAGES = DATA_DIR / "landmarks_images.csv"


def read_csv_raw(path: Path) -> Iterable[Tuple[str, np.ndarray]]:
    """Yield (label, 21x3 raw landmarks) for each row."""
    if not path.exists():
        return
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            label = row[0]
            pts = np.array(row[1:], dtype=np.float32).reshape(21, 3)
            yield label, pts


_WORKER_HANDS = None  # per-process MediaPipe Hands, initialized lazily in each worker


def _worker_init():
    import mediapipe as mp_lib
    global _WORKER_HANDS
    _WORKER_HANDS = mp_lib.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    )


def _worker_process(item: Tuple[str, str]):
    """Return (label, 63-long list of floats) or (label, None) if no hand detected."""
    import cv2
    label, img_path = item
    img = cv2.imread(img_path)
    if img is None:
        return label, None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = _WORKER_HANDS.process(rgb)
    if not res.multi_hand_landmarks:
        return label, None
    lms = res.multi_hand_landmarks[0].landmark
    floats = []
    for lm in lms:
        floats.extend([lm.x, lm.y, lm.z])
    return label, floats


def extract_landmarks_from_images(
    folder: Path, samples_per_class: int | None, out_csv: Path, workers: int | None = None
) -> None:
    """Run MediaPipe on images in parallel, cache (label, 63 floats) rows to out_csv."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted(p for p in folder.iterdir() if p.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class subdirectories in {folder}")

    jobs: List[Tuple[str, str]] = []
    per_class_totals: dict[str, int] = {}
    for class_dir in class_dirs:
        label = class_dir.name
        imgs = sorted(class_dir.glob("*.jp*g")) + sorted(class_dir.glob("*.png"))
        if samples_per_class and len(imgs) > samples_per_class:
            random.seed(42)
            imgs = random.sample(imgs, samples_per_class)
        per_class_totals[label] = len(imgs)
        for p in imgs:
            jobs.append((label, str(p)))

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"Extracting landmarks from {len(jobs)} images across {len(class_dirs)} classes "
          f"using {workers} workers")

    kept: dict[str, int] = {lbl: 0 for lbl in per_class_totals}
    skipped: dict[str, int] = {lbl: 0 for lbl in per_class_totals}

    t0 = time.time()
    done = 0
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        with mp.Pool(processes=workers, initializer=_worker_init) as pool:
            for label, floats in pool.imap_unordered(_worker_process, jobs, chunksize=8):
                done += 1
                if floats is None:
                    skipped[label] += 1
                else:
                    row = [label] + [f"{v:.6f}" for v in floats]
                    writer.writerow(row)
                    kept[label] += 1
                if done % 250 == 0 or done == len(jobs):
                    rate = done / max(time.time() - t0, 1e-6)
                    eta = (len(jobs) - done) / max(rate, 1e-6)
                    print(f"  {done}/{len(jobs)}  ({rate:.0f} img/s, ETA {eta:.0f}s)")

    total_written = sum(kept.values())
    print()
    for label in sorted(kept):
        print(f"  {label:>7}: kept {kept[label]:>4}, skipped {skipped[label]:>4}")
    elapsed = time.time() - t0
    print(f"Cached {total_written} landmark rows to {out_csv} in {elapsed:.1f}s")


NOISE_SIGMA = 0.003            # Gaussian noise on raw landmark coords (~0.3% of image)
ROT_DEG_RANGE = 8.0            # Small in-plane rotation; bigger would confuse G/Q, P/K


def _jitter(pts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply small z-axis rotation around image center + Gaussian landmark noise.
    Teaches robustness to real-world webcam jitter that the Kaggle studio photos
    don't contain, without changing the sign's meaning."""
    angle = rng.uniform(-ROT_DEG_RANGE, ROT_DEG_RANGE) * np.pi / 180.0
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    # Rotate around image center (0.5, 0.5) so normalized [0,1] coords stay sensible.
    out = pts.copy().astype(np.float32)
    out[:, :2] -= 0.5
    out = out @ rot.T
    out[:, :2] += 0.5
    out += rng.normal(0.0, NOISE_SIGMA, out.shape).astype(np.float32)
    return out


def load_features(
    skip_labels: set[str], mirror_augment: bool = True, jitter_augment: bool = True,
) -> tuple[np.ndarray, List[str], np.ndarray]:
    """Returns (X, y, is_user_sample). Augmentations, applied in order:
    - mirror_augment: each raw sample produces an additional horizontally-flipped
      copy (teaches left/right-hand invariance).
    - jitter_augment: each resulting sample also produces one noisy+mildly-rotated
      copy (teaches tolerance to webcam jitter + wrist tilt).
    Total expansion: 2× (mirror) × 2× (jitter) = 4× the raw row count."""
    rng = np.random.default_rng(42)
    X, y, is_user = [], [], []
    for src, from_user in ((CSV_CAPTURE, True), (CSV_IMAGES, False)):
        if not src.exists():
            continue
        raw_count = 0
        for label, pts in read_csv_raw(src):
            if label in skip_labels:
                continue
            variants = [pts]
            if mirror_augment:
                variants.append(mirror_landmarks(pts))
            for v in variants:
                X.append(build_features(v))
                y.append(label)
                is_user.append(from_user)
                if jitter_augment:
                    X.append(build_features(_jitter(v, rng)))
                    y.append(label)
                    is_user.append(from_user)
            raw_count += 1
        tag = "user capture — will be weighted up" if from_user else "Kaggle"
        factor = (2 if mirror_augment else 1) * (2 if jitter_augment else 1)
        aug_desc = []
        if mirror_augment:
            aug_desc.append("mirror")
        if jitter_augment:
            aug_desc.append("jitter")
        desc = "+".join(aug_desc) or "none"
        print(f"Loaded {raw_count} rows from {src.name} ({tag}); "
              f"augmented {factor}× to {raw_count * factor} via {desc}")
    if not X:
        raise SystemExit(
            "No training data found. Run capture.py or pass --images PATH."
        )
    return np.array(X, dtype=np.float32), y, np.array(is_user, dtype=bool)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=None,
                        help="Folder like data/asl_alphabet_train/<LABEL>/*.jpg")
    parser.add_argument("--samples-per-class", type=int, default=300,
                        help="Max images sampled per class during extraction")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers for extraction (default: cpu_count - 1)")
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--user-weight", type=float, default=50.0,
                        help="How many 'Kaggle samples' one of your own captures "
                             "counts as. 50 means 1 user sample ≈ 50 Kaggle samples.")
    parser.add_argument("--no-mirror-augment", action="store_true",
                        help="Disable horizontal-flip augmentation. Off by default; "
                             "enable this flag to train single-handedness only.")
    parser.add_argument("--no-jitter-augment", action="store_true",
                        help="Disable noise+micro-rotation augmentation. Off by default.")
    parser.add_argument("--skip-labels", type=str, default="",
                        help="Comma-separated labels to exclude (e.g. nothing,space,del)")
    parser.add_argument("--rebuild-image-cache", action="store_true",
                        help="Re-extract landmarks from --images even if cache exists")
    args = parser.parse_args()

    if args.images:
        if args.rebuild_image_cache or not CSV_IMAGES.exists():
            print(f"Extracting landmarks from {args.images} (max {args.samples_per_class}/class)")
            extract_landmarks_from_images(
                args.images, args.samples_per_class, CSV_IMAGES, workers=args.workers
            )
        else:
            print(f"Using cached image landmarks at {CSV_IMAGES} "
                  "(pass --rebuild-image-cache to regenerate)")

    skip = {s.strip() for s in args.skip_labels.split(",") if s.strip()}
    X, y, is_user = load_features(
        skip,
        mirror_augment=not args.no_mirror_augment,
        jitter_augment=not args.no_jitter_augment,
    )

    labels = sorted(set(y))
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    y_idx = np.array([label_to_idx[lbl] for lbl in y])
    n_user = int(is_user.sum())
    print(f"Training on {len(X)} samples across {len(labels)} classes "
          f"({n_user} from you, {len(X) - n_user} from Kaggle)")

    # Per-class user weight budget: every user class contributes the same total
    # weight (= user_weight × mean user-count per class). Classes with lots of
    # captures get a smaller per-sample weight; rare classes get boosted. Prevents
    # over-represented classes from tilting the decision boundary.
    sample_weight = np.ones(len(X), dtype=np.float32)
    if n_user:
        user_counts: dict[str, int] = {}
        for lbl, is_u in zip(y, is_user):
            if is_u:
                user_counts[lbl] = user_counts.get(lbl, 0) + 1
        target = sum(user_counts.values()) / len(user_counts)
        class_budget = args.user_weight * target
        for i, (lbl, is_u) in enumerate(zip(y, is_user)):
            if is_u:
                sample_weight[i] = class_budget / user_counts[lbl]
        print(f"Balancing {n_user} user samples across {len(user_counts)} classes: "
              f"each class gets total weight {class_budget:.0f} "
              f"(mean {target:.1f} samples × {args.user_weight}×)")
        min_lbl = min(user_counts, key=user_counts.get)
        max_lbl = max(user_counts, key=user_counts.get)
        print(f"  fewest: {min_lbl}={user_counts[min_lbl]} "
              f"(per-sample weight {class_budget / user_counts[min_lbl]:.1f}) | "
              f"most: {max_lbl}={user_counts[max_lbl]} "
              f"(per-sample weight {class_budget / user_counts[max_lbl]:.1f})")

    X_train, X_test, y_train, y_test, w_train, _w_test = train_test_split(
        X, y_idx, sample_weight, test_size=0.2, random_state=42, stratify=y_idx
    )

    clf = RandomForestClassifier(n_estimators=args.trees, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train, sample_weight=w_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=labels))

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": clf, "labels": labels, "feature_version": FEATURE_VERSION},
        MODEL_OUT,
    )
    print(f"Saved model to {MODEL_OUT} (feature_version={FEATURE_VERSION})")


if __name__ == "__main__":
    main()
