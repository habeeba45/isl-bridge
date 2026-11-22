"""SYNTHETIC landmark clips — for testing the evaluation pipeline only.

These are hand-made stick-figure hands, not recordings of people. Numbers
computed on them say nothing about real recognition accuracy and must never
be reported as results.

    python -m evaluation.synthetic --out /tmp/synthetic.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random

# Finger bases (MCP) relative to the wrist for an upright hand, image coords
# (y grows downwards), in units of hand size.
_MCP = {1: (-0.30, -0.85), 2: (-0.10, -1.0), 3: (0.10, -1.0), 4: (0.28, -0.90)}
_SEG = [0.40, 0.25, 0.22]


def _hand(fingers_up, scale, angle, cx, cy, rng, noise):
    """21 landmarks for a hand with given extended fingers [thumb..pinky]."""
    pts = [(0.0, 0.0)] * 21
    pts[0] = (0.0, 0.0)
    # Thumb: CMC 1, MCP 2, IP 3, TIP 4.
    pts[1] = (-0.25, -0.25)
    pts[2] = (-0.45, -0.45)
    if fingers_up[0]:
        pts[3], pts[4] = (-0.70, -0.65), (-0.92, -0.82)
    else:
        pts[3], pts[4] = (-0.35, -0.70), (-0.12, -0.78)
    for f in range(1, 5):
        bx, by = _MCP[f]
        mcp, pip, dip, tip = 1 + 4 * f, 2 + 4 * f, 3 + 4 * f, 4 + 4 * f
        pts[mcp] = (bx, by)
        if fingers_up[f]:
            y = by
            pts[pip] = (bx, y - _SEG[0])
            pts[dip] = (bx, y - _SEG[0] - _SEG[1])
            pts[tip] = (bx, y - sum(_SEG))
        else:
            pts[pip] = (bx, by - 0.30)
            pts[dip] = (bx, by - 0.12)
            pts[tip] = (bx, by + 0.05)
    ca, sa = math.cos(angle), math.sin(angle)
    out = []
    for x, y in pts:
        rx, ry = x * ca - y * sa, x * sa + y * ca
        out.append({
            "x": round(cx + rx * scale + rng.gauss(0, noise), 4),
            "y": round(cy + ry * scale + rng.gauss(0, noise), 4),
            "z": round(rng.gauss(0, noise), 4),
        })
    return out


PALM = [True, True, True, True, True]
FIST = [False, False, False, False, False]
SHAPES = {
    "HELLO": PALM, "PLEASE": PALM, "THANK_YOU": PALM, "STOP": PALM,
    "YES": FIST, "NO": [False, True, False, False, False],
    "NAME": [False, True, True, False, False],
    "WATER": [False, True, True, True, False],
    "GOOD": [True, False, False, False, False],
}
NEG_SHAPES = [
    [False, True, False, False, True], [False, False, True, False, False],
    [True, False, False, False, True], [True, True, False, False, True],
    [False, False, False, False, True],
]


def _motion(label, t, amp, freq, phase):
    w = 2 * math.pi * freq * t + phase
    if label in ("HELLO", "NO"):
        return amp * math.sin(w), 0.0
    if label == "PLEASE":
        return 0.6 * amp * math.cos(w), 0.6 * amp * math.sin(w)
    if label == "YES":
        return 0.0, 0.7 * amp * math.sin(w)
    if label == "NAME":
        return 0.0, 0.5 * amp * math.sin(w)
    if label == "THANK_YOU":
        return 0.0, 1.6 * amp * t
    return 0.0, 0.0


def make_clip(label, signer_rng, rng, n=75):
    s = signer_rng
    scale = 0.11 * s["size"] * rng.uniform(0.9, 1.1)
    angle = s["tilt"] + rng.gauss(0, 0.12)
    cx = 0.5 + rng.uniform(-0.08, 0.08)
    cy = (0.38 if label == "THANK_YOU" else 0.62) + rng.uniform(-0.05, 0.05)
    amp = s["amp"] * rng.uniform(0.75, 1.25)
    freq = rng.uniform(0.9, 1.6)
    phase = rng.uniform(0, 2 * math.pi)
    noise = s["noise"]
    neg_shape = rng.choice(NEG_SHAPES)
    neg_motion = rng.choice(["HELLO", "PLEASE", "YES", "STOP"])
    frames = []
    for i in range(n):
        t = i / (n - 1)
        if rng.random() < s["dropout"]:
            frames.append({"hands": []})
            continue
        if label == "HELP":
            a = _hand(FIST, scale, angle, cx - 0.05, cy - 0.04, rng, noise)
            b = _hand(PALM, scale, angle, cx + 0.05, cy + 0.02, rng, noise)
            frames.append({"hands": [a, b]})
            continue
        if label == "NEGATIVE":
            shape, mlabel = neg_shape, neg_motion
        else:
            shape, mlabel = SHAPES[label], label
        dx, dy = _motion(mlabel, t, amp, freq, phase)
        frames.append({"hands": [_hand(shape, scale, angle, cx + dx, cy + dy, rng, noise)]})
    return frames


def generate(n_signers=4, per_label=12, seed=0):
    rng = random.Random(seed)
    labels = list(SHAPES) + ["HELP", "NEGATIVE"]
    samples = []
    for si in range(n_signers):
        signer = {
            "size": rng.uniform(0.85, 1.2), "tilt": rng.gauss(0, 0.15),
            "amp": rng.uniform(0.06, 0.11), "noise": rng.uniform(0.002, 0.006),
            "dropout": rng.uniform(0.0, 0.08),
        }
        for label in labels:
            for k in range(per_label):
                samples.append({
                    "id": f"syn-{si}-{label}-{k}", "label": label,
                    "signer": f"SYN{si + 1}", "lighting": "synthetic",
                    "synthetic": True, "frames": make_clip(label, signer, rng),
                })
    return samples


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--signers", type=int, default=4)
    ap.add_argument("--per-label", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    with open(a.out, "w", encoding="utf-8") as fh:
        for s in generate(a.signers, a.per_label, a.seed):
            fh.write(json.dumps(s) + "\n")
    print(f"Wrote SYNTHETIC data to {a.out} (pipeline testing only)")
