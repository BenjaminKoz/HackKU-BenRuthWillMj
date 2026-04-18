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
import random
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]
MODEL_OUT = ROOT / "models" / "asl_classifier.joblib"
DATA_DIR = ROOT.parent / "data"
CSV_CAPTURE = DATA_DIR / "landmarks.csv"
CSV_IMAGES = DATA_DIR / "landmarks_images.csv"


def normalize(points: np.ndarray) -> np.ndarray:
    wrist = points[0].copy()
    pts = points - wrist
    scale = float(np.linalg.norm(pts[9])) or 1.0
    return (pts / scale).flatten()


def read_csv(path: Path) -> Iterable[Tuple[str, np.ndarray]]:
    if not path.exists():
        return
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            label = row[0]
            pts = np.array(row[1:], dtype=np.float32).reshape(21, 3)
            yield label, normalize(pts)


def extract_landmarks_from_images(
    folder: Path, samples_per_class: int | None, out_csv: Path
) -> None:
    """Run MediaPipe on images, cache (label, 63 floats) rows to out_csv."""
    import cv2
    import mediapipe as mp

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                     min_detection_confidence=0.5)

    class_dirs = sorted(p for p in folder.iterdir() if p.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class subdirectories in {folder}")

    total_written = 0
    t0 = time.time()
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        for class_dir in class_dirs:
            label = class_dir.name
            imgs = sorted(class_dir.glob("*.jp*g")) + sorted(class_dir.glob("*.png"))
            if samples_per_class and len(imgs) > samples_per_class:
                random.seed(42)
                imgs = random.sample(imgs, samples_per_class)

            class_written = 0
            class_skipped = 0
            for i, img_path in enumerate(imgs):
                img = cv2.imread(str(img_path))
                if img is None:
                    class_skipped += 1
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)
                if not res.multi_hand_landmarks:
                    class_skipped += 1
                    continue
                lms = res.multi_hand_landmarks[0].landmark
                row = [label]
                for lm in lms:
                    row.extend([f"{lm.x:.6f}", f"{lm.y:.6f}", f"{lm.z:.6f}"])
                writer.writerow(row)
                class_written += 1
            print(f"  {label:>7}: kept {class_written}, skipped {class_skipped} (no hand)")
            total_written += class_written

    elapsed = time.time() - t0
    print(f"Cached {total_written} landmark rows to {out_csv} in {elapsed:.1f}s")


def load_features(skip_labels: set[str]) -> tuple[np.ndarray, List[str]]:
    X, y = [], []
    for src in (CSV_CAPTURE, CSV_IMAGES):
        if not src.exists():
            continue
        count = 0
        for label, feats in read_csv(src):
            if label in skip_labels:
                continue
            X.append(feats)
            y.append(label)
            count += 1
        print(f"Loaded {count} rows from {src.name}")
    if not X:
        raise SystemExit(
            "No training data found. Run capture.py or pass --images PATH."
        )
    return np.array(X, dtype=np.float32), y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=None,
                        help="Folder like data/asl_alphabet_train/<LABEL>/*.jpg")
    parser.add_argument("--samples-per-class", type=int, default=300,
                        help="Max images sampled per class during extraction")
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--skip-labels", type=str, default="",
                        help="Comma-separated labels to exclude (e.g. nothing,space,del)")
    parser.add_argument("--rebuild-image-cache", action="store_true",
                        help="Re-extract landmarks from --images even if cache exists")
    args = parser.parse_args()

    if args.images:
        if args.rebuild_image_cache or not CSV_IMAGES.exists():
            print(f"Extracting landmarks from {args.images} (max {args.samples_per_class}/class)")
            extract_landmarks_from_images(args.images, args.samples_per_class, CSV_IMAGES)
        else:
            print(f"Using cached image landmarks at {CSV_IMAGES} "
                  "(pass --rebuild-image-cache to regenerate)")

    skip = {s.strip() for s in args.skip_labels.split(",") if s.strip()}
    X, y = load_features(skip)

    labels = sorted(set(y))
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    y_idx = np.array([label_to_idx[lbl] for lbl in y])
    print(f"Training on {len(X)} samples across {len(labels)} classes: {labels}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_idx, test_size=0.2, random_state=42, stratify=y_idx
    )

    clf = RandomForestClassifier(n_estimators=args.trees, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=labels))

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "labels": labels}, MODEL_OUT)
    print(f"Saved model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
