"""Fixed-length features from a landmark clip, for the learned baseline.

A clip is the same structure the live app sends to /api/recognize:
    [{"hands": [[{x, y, z} * 21], ...]}, ...]

Features (all computed from MediaPipe landmarks only):
  * hand shape  – primary-hand landmarks translated to the wrist and scaled by
                  wrist→middle-MCP length, then mean and std over frames
                  (63 + 63). This removes position and hand size.
  * motion      – wrist-trajectory statistics in image coordinates, reusing
                  the rule-based recognizer's own trajectory features, plus
                  mean wrist height.
  * two hands   – fraction of frames with two hands, second-hand normalized
                  shape (63) and mean inter-wrist distance.
  * visibility  – fraction of frames in which any hand was detected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recognize import _trajectory_features  # noqa: E402

N_LM = 21
MOTION_KEYS = [
    "x_range", "y_range", "path_x", "path_y",
    "x_reversals", "y_reversals", "loop_area_ratio",
]


def _hand_array(hand) -> np.ndarray | None:
    if isinstance(hand, dict):
        hand = hand.get("landmarks", [])
    if not isinstance(hand, list) or len(hand) < N_LM:
        return None
    return np.array(
        [[p["x"], p["y"], p.get("z", 0.0)] for p in hand[:N_LM]], dtype=float
    )


def normalize_hand(arr: np.ndarray) -> np.ndarray:
    """Translate to wrist, scale by wrist→middle-finger-MCP length."""
    centered = arr - arr[0]
    scale = np.linalg.norm(centered[9, :2])
    if scale < 1e-6:
        scale = 1.0
    return (centered / scale).reshape(-1)


def clip_features(frames: list) -> np.ndarray:
    primary, secondary, wrist_xy, inter = [], [], [], []
    with_hand = 0
    for frame in frames:
        hands = [h for h in (_hand_array(h) for h in (frame.get("hands") or [])) if h is not None]
        if not hands:
            continue
        with_hand += 1
        primary.append(normalize_hand(hands[0]))
        wrist_xy.append(hands[0][0, :2])
        if len(hands) >= 2:
            secondary.append(normalize_hand(hands[1]))
            inter.append(np.linalg.norm(hands[0][0, :2] - hands[1][0, :2]))

    dim = N_LM * 3
    if not primary:
        shape_mean = np.zeros(dim)
        shape_std = np.zeros(dim)
        motion = np.zeros(len(MOTION_KEYS) + 3)
    else:
        P = np.stack(primary)
        shape_mean = P.mean(0)
        shape_std = P.std(0)
        W = np.stack(wrist_xy)
        traj = _trajectory_features(W[:, 0].tolist(), W[:, 1].tolist())
        motion = np.array(
            [traj[k] for k in MOTION_KEYS]
            + [W[:, 0].std(), W[:, 1].std(), W[:, 1].mean()]
        )

    n = max(len(frames), 1)
    two_ratio = len(secondary) / max(with_hand, 1)
    sec_mean = np.stack(secondary).mean(0) if secondary else np.zeros(dim)
    inter_mean = float(np.mean(inter)) if inter else 0.0

    return np.concatenate([
        shape_mean, shape_std, motion,
        [two_ratio], sec_mean, [inter_mean],
        [with_hand / n],
    ])
