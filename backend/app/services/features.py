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
       - 3-component palm normal vector. Distinguishes letters that share finger
         shape but differ in orientation: G (palm sideways) vs Q (palm down), and
         P (palm down) vs K (palm forward).
       - 4 thumb-tip-to-finger-MCP distances. Explicitly encodes thumb position,
         the key discriminator in the fist family (A/E/M/N/S/T) where the whole
         hand looks similar but the thumb sits against different knuckles.

Final feature length: 63 + 10 + 5 + 5 + 3 + 4 = 90.
"""
from __future__ import annotations

from itertools import combinations
from typing import List, Sequence

import numpy as np

FEATURE_VERSION = 4  # bump when the feature layout changes (static + motion)
WORD_FEATURE_VERSION = 1  # word-clip features only — independent of FEATURE_VERSION
                          # so changes to two-hand encoding don't invalidate the
                          # static letter / J-Z motion models.

# Each frame in word mode is encoded as TWO hand slots, sorted by wrist x
# (leftmost first). A missing hand is encoded as zeros — the model learns
# "slot 1 zero" as the marker for one-handed signs.
HANDS_PER_FRAME = 2
SHAPE_FEATURE_LEN = 90  # length of build_features() output, used by callers

TIPS = [4, 8, 12, 16, 20]          # thumb, index, middle, ring, pinky fingertips
PIPS = [3, 6, 10, 14, 18]          # PIP joints (knuckle closest to the tip)
WRIST = 0
MIDDLE_MCP = 9
INDEX_MCP = 5
RING_MCP = 13
PINKY_MCP = 17
THUMB_TIP = 4
FINGER_MCPS = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

TIP_PAIRS = list(combinations(TIPS, 2))  # 10 pairs


def _normalize(points: np.ndarray) -> np.ndarray:
    """Translate to wrist origin, scale by hand size. Returns (21, 3)."""
    wrist = points[WRIST].copy()
    pts = points - wrist
    scale = float(np.linalg.norm(pts[MIDDLE_MCP])) or 1.0
    return pts / scale


def mirror_landmarks(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Horizontally flip raw MediaPipe landmarks (x in [0, 1] image space).

    Used as a training-time data augmentation so the classifier learns both
    left- and right-hand orientations. Flipping x: x -> 1 - x; y and z are
    unchanged.
    """
    arr = np.asarray(points, dtype=np.float32).reshape(21, 3).copy()
    arr[:, 0] = 1.0 - arr[:, 0]
    return arr


def sort_hands_by_x(hands: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Sort detected hands left-to-right by wrist x. Stable slot order is the
    only invariant that lets training and inference produce comparable feature
    vectors, since MediaPipe's "Left/Right" handedness label is unreliable on
    a mirrored webcam view."""
    return sorted(hands, key=lambda h: float(h[WRIST, 0]))


def pad_hands_to_two(hands: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Return exactly 2 hand arrays, padding missing slots with all-zero
    21x3 arrays. Caller must have already sorted hands into canonical order."""
    out: List[np.ndarray] = list(hands[:2])
    while len(out) < HANDS_PER_FRAME:
        out.append(np.zeros((21, 3), dtype=np.float32))
    return out


def _is_zero_hand(hand: np.ndarray) -> bool:
    return bool(np.all(hand == 0))


def build_two_hand_shape_features(hands: Sequence[np.ndarray]) -> np.ndarray:
    """Per-frame two-hand shape vector of length 2 * SHAPE_FEATURE_LEN.

    `hands` must be exactly 2 hand arrays in canonical order (use
    `pad_hands_to_two` after sorting). A zero-padded slot becomes a zero
    feature block, so the model can use slot-emptiness as a signal for
    "one-handed sign in the other slot."
    """
    parts: List[np.ndarray] = []
    for h in hands:
        if _is_zero_hand(h):
            parts.append(np.zeros(SHAPE_FEATURE_LEN, dtype=np.float32))
        else:
            parts.append(build_features(h).astype(np.float32))
    return np.concatenate(parts)


def build_features(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return a 1-D feature vector of length 90."""
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

    # Palm normal: cross product of (wrist -> index-MCP) and (wrist -> pinky-MCP),
    # normalized to a unit vector. Encodes which way the palm faces.
    n = np.cross(pts[INDEX_MCP], pts[PINKY_MCP])
    n_mag = float(np.linalg.norm(n)) or 1.0
    palm_normal = (n / n_mag).astype(np.float32)  # 3

    thumb_to_mcp = np.array(
        [np.linalg.norm(pts[THUMB_TIP] - pts[mcp]) for mcp in FINGER_MCPS],
        dtype=np.float32,
    )  # 4

    return np.concatenate(
        [flat, tip_pair_dists, tip_to_wrist, pip_to_tip, palm_normal, thumb_to_mcp]
    )
