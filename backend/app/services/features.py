"""Shared landmark feature builder. Imported by both training and inference paths
so the classifier always sees the exact same feature vector.

Input: a 21x3 array of MediaPipe hand landmarks (x, y, z) in [0, 1] image-normalized
coords. We then:

  1. Translate so the wrist (landmark 0) is at the origin.
  2. Scale by the wrist -> middle-finger MCP distance (landmark 9) so the hand has
     size 1. Removes "hand close to camera" vs "hand far away" variance.
  3. Flatten the 63 coords.
  4. Append hand-crafted distances that directly encode signals the model would
     otherwise have to learn from scratch:
       - 10 pairwise distances between the 5 fingertips (thumb, index, middle,
         ring, pinky). The index-middle distance is the primary U/V discriminator.
       - 5 fingertip-to-wrist distances (how "extended" each finger is).
       - 5 PIP-joint-to-tip distances (how "bent" each finger is).

Final feature length: 63 + 10 + 5 + 5 = 83.
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np

FEATURE_VERSION = 2  # bump when the feature layout changes

TIPS = [4, 8, 12, 16, 20]          # thumb, index, middle, ring, pinky fingertips
PIPS = [3, 6, 10, 14, 18]          # PIP joints (knuckle closest to the tip)
WRIST = 0
MIDDLE_MCP = 9

TIP_PAIRS = list(combinations(TIPS, 2))  # 10 pairs


def _normalize(points: np.ndarray) -> np.ndarray:
    """Translate to wrist origin, scale by hand size. Returns (21, 3)."""
    wrist = points[WRIST].copy()
    pts = points - wrist
    scale = float(np.linalg.norm(pts[MIDDLE_MCP])) or 1.0
    return pts / scale


def build_features(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return a 1-D feature vector of length 83."""
    arr = np.asarray(points, dtype=np.float32).reshape(21, 3)
    pts = _normalize(arr)

    flat = pts.flatten()  # 63

    tip_pair_dists = np.array(
        [np.linalg.norm(pts[a] - pts[b]) for a, b in TIP_PAIRS],
        dtype=np.float32,
    )  # 10

    tip_to_wrist = np.array(
        [np.linalg.norm(pts[t]) for t in TIPS],  # wrist is origin after normalize
        dtype=np.float32,
    )  # 5

    pip_to_tip = np.array(
        [np.linalg.norm(pts[tip] - pts[pip]) for tip, pip in zip(TIPS, PIPS)],
        dtype=np.float32,
    )  # 5

    return np.concatenate([flat, tip_pair_dists, tip_to_wrist, pip_to_tip])
